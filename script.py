import os
os.environ['DGL_PREFETCHER_TIMEOUT'] = str(300)
import ssl
import numpy as np
import torch
import dgl
import copy
import argparse
# import time


from utils import _get_device, train, test_val
from data_utils import TemporalDataset
from sampler import TemporalSampler, BatchedTemporalEdgePredictionSampler
from blocked_tgnn import TGN

# import time
# from six.moves import urllib
# import numpy as np
# import pandas as pd
# import torch
# import dgl
# import copy
# import argparse
# import inspect
# from dgl.dataloading import Sampler
# from sklearn.metrics import average_precision_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20,help='epochs for training on entire dataset')
parser.add_argument("--n_neighbors", type=int, default=10,
                        help="number of neighbors while doing embedding")
parser.add_argument("--batch_size", type=int,
                        default=8192, help="Size of each batch")
parser.add_argument("--embedding_dim", type=int, default=100,
                        help="Embedding dim for link prediction")
parser.add_argument("--memory_dim", type=int, default=100,
                        help="dimension of memory")
parser.add_argument("--temporal_dim", type=int, default=100,
                        help="Temporal dimension for time encoding")
parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of heads for multihead attention mechanism")
parser.add_argument("--memory_updater", type=str, default='gru',
                        help="Recurrent unit for memory update")
parser.add_argument("--k_hop", type=int, default=1,
                        help="sampling k-hop neighborhood")
parser.add_argument("--not_use_memory", action="store_true", default=False,
                        help="Enable memory for TGN Model disable memory for TGN Model")



# parser.add_argument("--epochs", type=int, default=20,help='epochs for training on entire dataset')
parser.add_argument("--num_negative_samples", type=int, default=1,
                        help="number of negative samplers per positive samples")
# parser.add_argument("--fast_mode", action="store_true", default=False,
#                         help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained")
# parser.add_argument("--n_neighbors", type=int, default=10,
#                         help="number of neighbors while doing embedding")
# parser.add_argument("--batch_size", type=int,
#                         default=256, help="Size of each batch")
# # parser.add_argument("--fast_mode", action="store_true", default=False,
# #                         help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained")

# parser.add_argument("--embedding_dim", type=int, default=100,
#                         help="Embedding dim for link prediction")
# parser.add_argument("--memory_dim", type=int, default=100,
#                         help="dimension of memory")
# parser.add_argument("--temporal_dim", type=int, default=100,
#                         help="Temporal dimension for time encoding")
# parser.add_argument("--num_heads", type=int, default=8,
#                         help="Number of heads for multihead attention mechanism")
# parser.add_argument("--memory_updater", type=str, default='gru',
#                         help="Recurrent unit for memory update")
# parser.add_argument("--k_hop", type=int, default=1,
#                         help="sampling k-hop neighborhood")
# parser.add_argument("--not_use_memory", action="store_true", default=False,
#                         help="Enable memory for TGN Model disable memory for TGN Model")
# # parser.add_argument("--aggregator", type=str, default='last',
# #                         help="Aggregation method for memory update")

# # args = parser.parse_args()
# # args.epochs = 50


args = parser.parse_args(args=[])
data_file = "wikipedia"


if __name__ == "__main__":
    data =  TemporalDataset(data_file, force_reload=False)
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()
    TRAIN_SPLIT = 0.7
    VALID_SPLIT = 0.85

    # set random Seed
    np.random.seed(2021)
    torch.manual_seed(2021)

    trainval_div = int(VALID_SPLIT*num_edges)
    test_split_ts = data.edata['timestamp'][trainval_div]
    test_nodes = torch.cat([data.edges()[0][trainval_div:], data.edges()[1][trainval_div:]]).unique().numpy()
    test_new_nodes = np.random.choice(test_nodes, int(0.1*len(test_nodes)), replace=False)

    in_subg = dgl.in_subgraph(data, test_new_nodes)
    out_subg = dgl.out_subgraph(data, test_new_nodes)
    # Remove edge who happen before the test set to prevent from learning the connection info
    new_node_in_eid_delete = in_subg.edata[dgl.EID][in_subg.edata['timestamp'] < test_split_ts]
    #gets the eids in in_sub_g that has time_stamp lower than 
    new_node_out_eid_delete = out_subg.edata[dgl.EID][out_subg.edata['timestamp'] < test_split_ts]
    #all the inbound and outbound edges that occurred before test_split_ts
    new_node_eid_delete = torch.cat([new_node_in_eid_delete, new_node_out_eid_delete]).unique()
    graph_new_node = copy.deepcopy(data)
    graph_new_node.remove_edges(new_node_eid_delete)
    #Here we have removed only the edges that occur before test_split_ts and have node_id 
    #belonging to test_split new nodes. In this way there will be no 
    #edge in the train or val split these edges

    in_eid_delete = in_subg.edata[dgl.EID]
    out_eid_delete = out_subg.edata[dgl.EID]
    eid_delete = torch.cat([in_eid_delete, out_eid_delete]).unique()

    graph_no_new_node = copy.deepcopy(data)
    graph_no_new_node.remove_edges(eid_delete)

    # Set Train, validation, test and new node test id
    train_seed = torch.arange(int(TRAIN_SPLIT*graph_no_new_node.num_edges()))
    # print(train_seed)
    valid_seed = torch.arange(int(TRAIN_SPLIT*graph_no_new_node.num_edges()), trainval_div-new_node_eid_delete.size(0))
    # print(valid_seed)
    test_seed = torch.arange(trainval_div-new_node_eid_delete.size(0), graph_no_new_node.num_edges())
    # print(test_seed)
    test_new_node_seed = torch.arange(trainval_div-new_node_eid_delete.size(0), graph_new_node.num_edges())
    # print(test_new_node_seed)


    temporal_sampler = TemporalSampler(k=args.n_neighbors)
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(k=args.num_negative_samples)
    temporal_edge_sampler = BatchedTemporalEdgePredictionSampler(temporal_sampler,  negative_sampler=neg_sampler)



    device = _get_device()
    # sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    train_dataloader = dgl.dataloading.DataLoader(
        graph_no_new_node, train_seed, temporal_edge_sampler,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator

    # sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    valid_dataloader = dgl.dataloading.DataLoader(
        graph_no_new_node, valid_seed, temporal_edge_sampler,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator

    test_new_node_dataloader = dgl.dataloading.DataLoader(
        graph_new_node,test_new_node_seed, temporal_edge_sampler,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator

    test_dataloader = dgl.dataloading.DataLoader(
        graph_no_new_node, test_seed, temporal_edge_sampler,
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator


    edge_dim = data.edata['feats'].shape[1]
    num_node = data.num_nodes()
    model = TGN(edge_feat_dim=edge_dim,
                    memory_dim=args.memory_dim,
                    temporal_dim=args.temporal_dim,
                    embedding_dim=args.embedding_dim,
                    num_heads=args.num_heads,
                    num_nodes=num_node,
                    n_neighbors=args.n_neighbors,
                    memory_updater_type=args.memory_updater,
                    mem_device = device,
                    layers=args.k_hop)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    f = open("logging_a100_8k.txt", 'w')
    # if args.fast_mode:
    #     sampler.reset()
    # model.to(device)
    model = model.to(device=device)





# sampler = temporal_edge_sampler

# # print("At the beginning") Sampler.reset
# # i = -1
# # nn_test_ap, nn_test_auc = test_val(
# #                 model, test_new_node_dataloader, sampler, criterion, args)
# # test_ap, test_auc = test_val(
# #                 model, test_dataloader, sampler, criterion, args)
# # print("Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i, test_ap, test_auc))
# print("Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(
#                 i, nn_test_ap, nn_test_auc))

    try:
            for i in range(args.epochs):
                train_loss, dl_tt, tr_tt = train(model, train_dataloader, temporal_edge_sampler,
                                criterion, optimizer, args)
                # print("total")
                val_ap, val_auc = test_val(
                    model, valid_dataloader, temporal_edge_sampler, criterion, args)
                memory_checkpoint = model.store_memory()
                # # print(memory_checkpoint)
                # if args.fast_mode:
                #     new_node_sampler.sync(sampler)
                test_ap, test_auc = test_val(
                    model, test_dataloader, temporal_edge_sampler, criterion, args)
                model.restore_memory(memory_checkpoint)
                # # print("after restoring: ", model.memory.memory)
                # if args.fast_mode:
                #     sample_nn = new_node_sampler
                # else:
                # sample_nn = sampler
                nn_test_ap, nn_test_auc = test_val(
                    model, test_new_node_dataloader, temporal_edge_sampler, criterion, args)
                log_content = []
                log_content.append("Epoch: {}; Training Loss: {} | Validation AP: {:.3f} AUC: {:.3f}\n".format(
                    i, train_loss, val_ap, val_auc))
                log_content.append(
                    "Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i, test_ap, test_auc))
                log_content.append("Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(
                    i, nn_test_ap, nn_test_auc))
                log_content.append("total time: dataloading: {} and overall: {}\n".format(dl_tt, tr_tt))
                f.writelines(log_content)
                # # print("before reset: ", model.memory.memory)
                model.reset_memory()
                # # print("after reset: ", model.memory.memory)
                # if i < args.epochs-1:
                #     temporal_edge_sampler.reset()
                print(log_content[0], log_content[1], log_content[2])
    except KeyboardInterrupt:
            traceback.print_exc()
            error_content = "Training Interreputed!"
            f.writelines(error_content)
            f.close()
    # total_blk = 0
    # blk_cnt = 0
    # max_blk_len = -1
    print("Max block length: ", max_blk_len)
    print("Avg block length: ", total_blk/blk_cnt)
    print(args.batch_size)
    print("========Training is Done========")