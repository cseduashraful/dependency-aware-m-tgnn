import os
os.environ['DGL_PREFETCHER_TIMEOUT'] = str(300)
import ssl
import numpy as np
import torch
import dgl
import copy
import argparse
from functools import partial
# import time


from static_sampler import static_positive_sampler as sp, interleave_seed as ins
from static_sampler import static_positive_sampler_v2 as sp2
from utils import _get_device, train2 as train, test_val2 as test_val, profileModel
from data_utils import TemporalDataset, convert2tcsr
from new_sampler import TemporalSampler, BatchedTemporalEdgePredictionSampler, MergedSampler
from blocked_tgnn import TGN

from sampler_core import ParallelSampler, TemporalGraphBlock
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-b","--batch_size", type=int, default=16*1024, help="Size of each batch")
parser.add_argument("-d","--data", type=str, default="wikipedia", help="data file name")
parser.add_argument("-k","--n_neighbors", type=int, default=10, help="number of neighbors while doing embedding")
parser.add_argument("--epochs", type=int, default=100,help='epochs for training on entire dataset')
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
parser.add_argument("--num_negative_samples", type=int, default=1,
                        help="number of negative samplers per positive samples")



args = parser.parse_args()

def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param


if __name__ == "__main__":
    data =  TemporalDataset(args.data, force_reload=False, bipartite=True)

        # data =  TemporalDataset(data_file, force_reload=False)
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()
    TRAIN_SPLIT = 0.8
    VALID_SPLIT = 0.9

    # set random Seed
    np.random.seed(2021)
    torch.manual_seed(2021)


    # train_seed = torch.arange(int(TRAIN_SPLIT*data.num_edges()))
    # print(train_seed)


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

    # new_node_eid_delete = torch.tensor([], data)

    

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
    # print("ts shape: ", train_seed.shape)
    # valid_seed = torch.arange(int(TRAIN_SPLIT*graph_no_new_node.num_edges()), trainval_div-new_node_eid_delete.graphs, _ = dgl.load_graphs('graph.dgl')
    # print(valid_seed.shape)
    valid_seed = torch.arange(int(TRAIN_SPLIT*graph_no_new_node.num_edges()), trainval_div-new_node_eid_delete.size(0))
    test_seed = torch.arange(trainval_div-new_node_eid_delete.size(0), graph_no_new_node.num_edges())
    # print(test_seed.shape)
    test_new_node_seed = torch.arange(trainval_div-new_node_eid_delete.size(0), graph_new_node.num_edges())



    # print(graph_no_new_node.ndata[dgl.NID])
    basic_sampler = partial(dgl.sampling.select_topk, k=args.n_neighbors, weight='timestamp')
    train_batches  =  sp2(graph_no_new_node, args.batch_size , basic_sampler, seed = torch.arange(int(TRAIN_SPLIT*data.num_edges())))
    # train_batches  =  sp(data, args.batch_size, basic_sampler, seed = torch.arange(int(TRAIN_SPLIT*data.num_edges())))
    valid_batches = sp2(graph_no_new_node, args.batch_size, basic_sampler, seed = valid_seed)
    # print("t..")
    test_batches = sp2(graph_no_new_node, args.batch_size, basic_sampler, seed = test_seed)
    # print("t1..")
    test_new_node_batches = sp2(graph_new_node, args.batch_size, basic_sampler, seed = test_new_node_seed)


    train_seed = ins(train_seed, train_batches)
    # print(train_seed)
    valid_seed = ins(valid_seed, valid_batches)
    test_seed = ins(test_seed, test_batches)
    test_new_node_seed = ins(test_new_node_seed, test_new_node_batches)

    







    print("number of edges: ", graph_no_new_node.num_edges())
    # print("actual: ", train_batches)
    g = convert2tcsr(graph_no_new_node)

    sample_param, memory_param, gnn_param, train_param = parse_config("TGN.yml")

    tgl_sampler = None
# if not ('no_sample' in sample_param and sample_param['no_sample']):
    tgl_sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                              sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                              sample_param['strategy']=='recent', sample_param['prop_time'],
                              sample_param['history'], float(sample_param['duration']))






# from collections import Counter

# # Sample list of numbers
# numbers_list = train_batches[0]#[1, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 8, 8, 8, 8]

# # Count the frequency of each number in the list
# frequency_count = Counter(numbers_list)

# # Print the frequency of each number
# for number, count in frequency_count.items():
#     print(f"{number}: {count} times")
#     input()


    temporal_sampler = TemporalSampler(k=args.n_neighbors)
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(k=args.num_negative_samples)
    # temporal_edge_sampler = BatchedTemporalEdgePredictionSampler(temporal_sampler,  negative_sampler=neg_sampler)
    # ms = MergedSampler(tgl_sampler, dgl_sampler = temporal_edge_sampler, negative_sampler=neg_sampler)
    ms = MergedSampler(tgl_sampler, negative_sampler=neg_sampler)




        
    device = _get_device()
    # sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    # train_dataloader = dgl.dataloading.DataLoader(
    #     graph_no_new_node, train_seed, temporal_edge_sampler,
    #     batch_size=args.batch_size*2, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator

    train_dataloader2 = dgl.dataloading.DataLoader(
        graph_no_new_node, train_seed, ms,
        batch_size=args.batch_size*2, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator



    # sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    # valid_dataloader = dgl.dataloading.DataLoader(
    #     graph_no_new_node, valid_seed, temporal_edge_sampler,
    #     batch_size=args.batch_size*2, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator

    valid_dataloader2 = dgl.dataloading.DataLoader(
        graph_no_new_node, valid_seed, ms,
        batch_size=args.batch_size*2, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator



    # test_new_node_dataloader = dgl.dataloading.DataLoader(
    #     graph_new_node,test_new_node_seed, temporal_edge_sampler,
    #     batch_size=args.batch_size*2, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator


    test_new_node_dataloader2 = dgl.dataloading.DataLoader(
        graph_new_node,test_new_node_seed, ms,
        batch_size=args.batch_size*2, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator

    # test_dataloader = dgl.dataloading.DataLoader(
    #     graph_no_new_node, test_seed, temporal_edge_sampler,
    #     batch_size=args.batch_size*2, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator

    test_dataloader2 = dgl.dataloading.DataLoader(
        graph_no_new_node, test_seed, ms,
        batch_size=args.batch_size*2, shuffle=False, drop_last=False, num_workers=0, device=device)#collate_fn = edge_collator






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
    factor =  args.batch_size/600
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001*factor)
    log_name = args.data+"_"+str(args.batch_size)+"_log.txt"
    f = open(log_name, 'w')
    # if args.fast_mode:
    #     sampler.reset()
    # model.to(device)
    model = model.to(device=device)
    pm = input("Profile model? ")
    if pm == 'y':
        profileModel(model, train_dataloader2)
    input("Start training?")

    print("Training started")
    try:
            for i in range(args.epochs):
                train_loss, dl_tt, tr_tt = train(model, train_dataloader2, None,
                                criterion, optimizer, args)
                # print("total")
                val_ap, val_auc = test_val(
                    model, valid_dataloader2, None, criterion, args)
                memory_checkpoint = model.store_memory()

                test_ap, test_auc = test_val(
                    model, test_dataloader2, None, criterion, args)
                model.restore_memory(memory_checkpoint)

                nn_test_ap, nn_test_auc = test_val(
                    model, test_new_node_dataloader2, None, criterion, args)
                
                log_content = []
                log_content.append("Epoch: {}; Training Loss: {} | Validation AP: {:.3f} AUC: {:.3f}\n".format(
                    i, train_loss, val_ap, val_auc))
                log_content.append(
                    "Epoch: {}; Test AP: {:.3f} AUC: {:.3f}\n".format(i, test_ap, test_auc))
                log_content.append("Epoch: {}; Test New Node AP: {:.3f} AUC: {:.3f}\n".format(
                    i, nn_test_ap, nn_test_auc))
                log_content.append("total time: dataloading: {} and overall: {}\n".format(dl_tt, tr_tt))
                f.writelines(log_content)
                
                model.reset_memory()
                
                print(log_content[0], log_content[1], log_content[2], log_content[3])
    except KeyboardInterrupt:
            traceback.print_exc()
            error_content = "Training Interreputed!"
            f.writelines(error_content)
            f.close()
    # total_blk = 0
    # blk_cnt = 0
    # max_blk_len = -1
    # print("Max block length: ", max_blk_len)
    # print("Avg block length: ", total_blk/blk_cnt)
    # print(args.batch_size)
    print("========Training is Done========")