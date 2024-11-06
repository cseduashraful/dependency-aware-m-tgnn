import dgl
import torch
import numpy as np

def static_positive_sampler_v2(g, batch_size, sampler, seed = None, ignore_direction = False, add_self_loop = True):
    bi = 0
    n = g.num_edges()
    batches = []
    maximum = -1
    last_index = 0
    # print("start: ", seed)
    if seed is not None:
        last_index = seed[0].item()
        n = seed[-1]+1
    # print("end: ", seed)
    while last_index < n:
        # last_index = bi*num_batch
        fl = batch_size if n>=batch_size+last_index else  n-last_index
        batches.append(static_positive_sampler_batch_v2(g, last_index, fl, sampler, ignore_direction, add_self_loop))
        last_index += batch_size
    
    return batches    


def static_positive_sampler_batch_v2(g, last_index, batch_size, sampler, ignore_direction, add_self_loop):
    # print(batch_size)
    srcs, dsts = g.edges()
    # print(edges)
    # last_index = 0

    
    ts = g.edata['timestamp'][last_index:last_index+batch_size]
    src_b = srcs[last_index:last_index+batch_size]
    dst_b = dsts[last_index:last_index+batch_size]
    block_ids =  InserInBlock_v2(ts, src_b, dst_b)
    # print(block_ids)
    # input("sp ln 84: ")
    print(".... ", max(block_ids))
    # if(verifyBlock(block_ids)==False):
    #     print(block_ids)
    return block_ids


def InserInBlock_v2(tss, src_b, dst_b):
    last_accessed_dict = {}
    access_dict = {}
    t_access_dict = {}
    blocks = []
    block_ids = []
    # print("len subgs: ", len(subgs))
    for i, ts in enumerate(tss):
        # ts = subg.ndata["timestamp"][0]#tss[i]
        
        edge_nodes  = torch.tensor([src_b[i], dst_b[i]])#
        # print("edge nodes: ", edge_nodes)

        last_accessed_times = [last_accessed_dict[k.item()] if k.item() in last_accessed_dict.keys() else -1 for k in edge_nodes]

        # src, dst = subg.edges()
        # edge_nodes = torch.unique(torch.cat((src, dst),0))
        # last_accessed_times = [last_accessed_dict[k.item()] if k.item() in last_accessed_dict.keys() else -1 for k in subg.ndata[dgl.NID][edge_nodes]]

        # print(last_accessed_times)
        # print(subg.ndata[dgl.NID])
        block_id = max(last_accessed_times)+1 if len(last_accessed_times)>0 else 0
        
        # block_id = min(max(last_accessed_times)+1 if len(last_accessed_times)>0 else 0, 0)
        # blocks.append([subg])if len(blocks)<=block_id else blocks[block_id].append(subg)
        for k in edge_nodes:
        # for k in subg.ndata[dgl.NID]:
            last_accessed_dict[k.item()] =  block_id
            kitem = k.item()
            if k.item() in access_dict.keys():
                # print(access_dict[kitem])
                access_dict[kitem].append(block_id)
                t_access_dict[k.item()].append(ts)
            else:
                access_dict[kitem] = [block_id]
                t_access_dict[k.item()] = [ts]    
        block_ids.append(block_id)
    return block_ids




def static_positive_sampler(g, batch_size, sampler, seed = None, ignore_direction = False, add_self_loop = True):
    #independent
    bi = 0
    n = g.num_edges()
    batches = []
    maximum = -1
    last_index = 0
    # print("start: ", seed)
    if seed is not None:
        last_index = seed[0].item()
        n = seed[-1]+1
    # print("end: ", seed)
    while last_index < n:
        # last_index = bi*num_batch
        fl = batch_size if n>=batch_size+last_index else  n-last_index
        batches.append(static_positive_sampler_batch(g, last_index, fl, sampler, ignore_direction, add_self_loop))
        last_index += batch_size
    
    # for i, batch in enumerate(batches):
    #     tmp = max(batch)
    #     # print("batch: ", i+1, " Req Block: ", tmp)
    #     if(tmp>maximum):
    #         maximum = tmp
    
    # print("Longest batch: ", maximum)
    
    return batches




def static_positive_sampler_batch(g, last_index, batch_size, sampler, ignore_direction, add_self_loop):
    # print(batch_size)
    srcs, dsts = g.edges()
    # print(edges)
    # last_index = 0

    
    ts = g.edata['timestamp'][last_index:last_index+batch_size]
    src_b = srcs[last_index:last_index+batch_size]
    dst_b = dsts[last_index:last_index+batch_size]

    # print("srcs: ", srcs)
    # print("srcb: ", src_b)
    # seed_nodes = torch.cat((src_b, dst_b), 0)
    # seed_ts = torch.cat((ts, ts), 0)

    # print("Seed nodes: ", seed_nodes)
    # print(seed_ts)

    subgs = []

    ##need vectorized implementation of in_subgs upto currect time_stamps
    ##independent, multi-threaded implementation possible
    for i, s_node in enumerate(src_b):
        d_node = dst_b[i]
        seed_nodes = [s_node, d_node]
        full_neighbor_subgraph = dgl.in_subgraph(g, seed_nodes)
        if ignore_direction:
            out_neighbor_subgraph = dgl.out_subgraph(g, seed_nodes)
            all_eids = torch.unique(torch.cat((full_neighbor_subgraph.edata["_ID"], out_neighbor_subgraph.edata["_ID"]),0))
            full_neighbor_subgraph = dgl.edge_subgraph(g, all_eids)
        if add_self_loop:
            full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,seed_nodes, seed_nodes)
        temporal_edge_mask = (full_neighbor_subgraph.edata['timestamp'] < ts[i]) + (full_neighbor_subgraph.edata['timestamp'] <= 0)
        temporal_subgraph = dgl.edge_subgraph(full_neighbor_subgraph, temporal_edge_mask)
        temp2origin = temporal_subgraph.ndata[dgl.NID]
        root2sub_dict = dict(zip(temp2origin.tolist(), temporal_subgraph.nodes().tolist()))
        seed_nodes = [root2sub_dict[int(n)] for n in seed_nodes]
        final_subgraph = sampler(g=temporal_subgraph, nodes=seed_nodes)
        final_subgraph.remove_self_loop()
        final_subgraph.ndata['timestamp'] = ts[i].repeat(final_subgraph.num_nodes())
        subgs.append(final_subgraph)
        # src_nodes, dest_nodes =  final_subgraph.edges()
        # src_nodes = src_nodes.unique()
        # dest_nodes = dest_nodes.unique()

    #dependent
    # print("len subgs: ", len(subgs))
    block_ids =  InserInBlock(subgs, ts, src_b, dst_b)
    # print(block_ids)
    # input("sp ln 84: ")
    print(".... ", max(block_ids))
    # if(verifyBlock(block_ids)==False):
    #     print(block_ids)
    return block_ids



def InserInBlock(subgs, tss, src_b, dst_b):
    last_accessed_dict = {}
    access_dict = {}
    t_access_dict = {}
    blocks = []
    block_ids = []
    # print("len subgs: ", len(subgs))
    for i, subg in enumerate(subgs):
        ts = subg.ndata["timestamp"][0]#tss[i]
        
        edge_nodes  = torch.tensor([src_b[i], dst_b[i]])#
        # print("edge nodes: ", edge_nodes)

        last_accessed_times = [last_accessed_dict[k.item()] if k.item() in last_accessed_dict.keys() else -1 for k in edge_nodes]

        # src, dst = subg.edges()
        # edge_nodes = torch.unique(torch.cat((src, dst),0))
        # last_accessed_times = [last_accessed_dict[k.item()] if k.item() in last_accessed_dict.keys() else -1 for k in subg.ndata[dgl.NID][edge_nodes]]

        # print(last_accessed_times)
        # print(subg.ndata[dgl.NID])

        block_id = max(last_accessed_times)+1 if len(last_accessed_times)>0 else 0
        blocks.append([subg])if len(blocks)<=block_id else blocks[block_id].append(subg)
        for k in edge_nodes:
        # for k in subg.ndata[dgl.NID]:
            last_accessed_dict[k.item()] =  block_id
            kitem = k.item()
            if k.item() in access_dict.keys():
                # print(access_dict[kitem])
                access_dict[kitem].append(block_id)
                t_access_dict[k.item()].append(ts)
            else:
                access_dict[kitem] = [block_id]
                t_access_dict[k.item()] = [ts]    
        block_ids.append(block_id)
    return block_ids


def interleave_seed(seed_edge, block_ids):
    batch_size = len(block_ids[0])
    idx = 0
    updated_seed = torch.tensor([], dtype=int)
    while idx<len(block_ids):
        updated_seed = torch.cat((updated_seed, seed_edge[idx*batch_size:(idx+1)*batch_size],torch.tensor(block_ids[idx])),0)
        idx += 1

    # print("updated seed: ", updated_seed[0:32])
    return updated_seed


def verifyBlock(arr):
    # print("arr: ", arr)
    maximum = max(arr)
    freq = [0 for i in range(maximum+1)]
    for i in arr:
        freq[i] += 1
    # print("freq: ", freq)
    for i in freq:
        if i == 0:
            return False
    return True