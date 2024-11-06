import torch
import random
import dgl
from dgl.dataloading import BlockSampler, Sampler, Mapping, EID, NID, F, context_of, heterograph, find_exclude_eids, compact_graphs, set_edge_lazy_features
from functools import partial
import inspect

import numpy as np

from static_sampler import verifyBlock as vb

from isomorphism import *
# from dgl.dataloading import EdgePredictionSampler, set_edge_lazy_features, compact_graphs, Mapping, EID, NID, F, context_of, heterograph, find_exclude_eids

class TemporalSampler(BlockSampler):
    """ Temporal Sampler builds computational and temporal dependency of node representations via
    temporal neighbors selection and screening.

    The sampler expects input node to have same time stamps, in the case of TGN, it should be
    either positive [src,dst] pair or negative samples. It will first take in-subgraph of seed
    nodes and then screening out edges which happen after that timestamp. Finally it will sample
    a fixed number of neighbor edges using random or topk sampling.

    Parameters
    ----------
    sampler_type : str
        sampler indication string of the final sampler.

        If 'topk' then sample topk most recent nodes

        If 'uniform' then uniform randomly sample k nodes

    k : int
        maximum number of neighors to sampler

        default 10 neighbors as paper stated

    Examples
    ----------
    Please refers to examples/pytorch/tgn/train.py

    """

    def __init__(self, sampler_type='topk', k=10):
        super(TemporalSampler, self).__init__()
        if sampler_type == 'topk':
            self.sampler = partial(
                dgl.sampling.select_topk, k=k, weight='timestamp')
        elif sampler_type == 'uniform':
            self.sampler = partial(dgl.sampling.sample_neighbors, fanout=k)
        else:
            raise DGLError(
                "Sampler string invalid please use \'topk\' or \'uniform\'")

    def sample(
        self, g, seed_nodes, exclude_eids=None, timestamp = 0,
    ):  # pylint: disable=arguments-differ
        """Sample a list of blocks from the given seed nodes."""
        result = self.sample_blocks(g, seed_nodes, timestamp, exclude_eids=exclude_eids)
        return self.assign_lazy_features(result)
    
    def sampler_frontier(self,
                         block_id,
                         g,
                         seed_nodes,
                         timestamp):
        full_neighbor_subgraph = dgl.in_subgraph(g, seed_nodes)
        # print("Full NSG1: ", full_neighbor_subgraph)
        # print("seed nodes: ", seed_nodes)
        
        # Adding self loops? but why?
        full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,
                                               seed_nodes, seed_nodes)
        # print("full_neighbor_subgraph edges: ", full_neighbor_subgraph.edges())
        # print("Full NSG2: ", full_neighbor_subgraph)
        
        #Remove edges that occurs after timetsamp
        temporal_edge_mask = (full_neighbor_subgraph.edata['timestamp'] < timestamp) + (
            full_neighbor_subgraph.edata['timestamp'] <= 0)
        # print("temporal_edge_mask: ", temporal_edge_mask)
        temporal_subgraph = dgl.edge_subgraph(
            full_neighbor_subgraph, temporal_edge_mask)

        # Map preserve ID
        temp2origin = temporal_subgraph.ndata[dgl.NID]
#         print("seed nodes: ", seed_nodes)
#         print("temp2origin (actual node ids): ", temp2origin)
#         print("temporal_subgraph edges: ", temporal_subgraph.edges())
#         print("temporal subgraph nids: ", temporal_subgraph.ndata[dgl.NID])
        

        # The added new edgge will be preserved hence
        root2sub_dict = dict(
            zip(temp2origin.tolist(), temporal_subgraph.nodes().tolist()))
        
        
        # temporal_subgraph.ndata["_orgID"] = g.ndata[dgl.NID][temp2origin]
        # print("temporal subgraph nids: ", temporal_subgraph.ndata[dgl.NID])
        # print("seed node before updates: ", seed_nodes)
        seed_nodes = [root2sub_dict[int(n)] for n in seed_nodes]
        
        # print("updated seed nodes: ", seed_nodes)
        # print("")
        final_subgraph = self.sampler(g=temporal_subgraph, nodes=seed_nodes)
        final_subgraph.remove_self_loop()
        # print("final subgraphs: ", final_subgraph)
        # print("Eids:", final_subgraph.edges())
        # print("Nids:", final_subgraph.nodes())
        # print("ndata: ", final_subgraph.ndata)
        src_nodes, dest_nodes =  final_subgraph.edges()
        src_nodes = src_nodes.unique()
        dest_nodes = dest_nodes.unique()
        # print("s: ", src_nodes)
        # print("d: ", dest_nodes)
        block = dgl.transforms.to_block(final_subgraph, seed_nodes)
        block2 = dgl.transforms.to_block(final_subgraph)
        
#         print("block: ", block)
#         print("block ndata: ", block.ndata)
#         print("block srcdata: ", block.srcdata)
#         print("block dstdata: ", block.dstdata)
        
        # print("block2: ", block2)
        # print("block2 ndata: ", block2.ndata)
        # print("block2 srcdata: ", block2.srcdata)
        # print("block2 dstdata: ", block2.dstdata)
        return src_nodes, dest_nodes, block #dgl.transforms.to_block(final_subgraph)

    
    
    def sampler_frontier_for_Batch(self,
                         block_id,
                         g,
                         seed_nodes,
                         timestamp):
        
        full_neighbor_subgraph = dgl.in_subgraph(g, seed_nodes)
        # print("Full NSG1: ", full_neighbor_subgraph)
        # print("seed nodes: ", seed_nodes)
        
        # Adding self loops? but why?

        # unique_list = list(set(seed_nodes))
        # full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,
        #                                        unique_list, unique_list)
        full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,
                                               seed_nodes, seed_nodes)
                                        
        # print("full_neighbor_subgraph edges: ", full_neighbor_subgraph.edges())
        # print("Full NSG2: ", full_neighbor_subgraph)
        
        #Remove edges that occurs after timetsamp
        temporal_edge_mask = (full_neighbor_subgraph.edata['timestamp'] < timestamp) + (
            full_neighbor_subgraph.edata['timestamp'] <= 0)
        # print("temporal_edge_mask: ", temporal_edge_mask)
        temporal_subgraph = dgl.edge_subgraph(
            full_neighbor_subgraph, temporal_edge_mask)

        # Map preserve ID
        temp2origin = temporal_subgraph.ndata[dgl.NID]
#         print("seed nodes: ", seed_nodes)
#         print("temp2origin (actual node ids): ", temp2origin)
#         print("temporal_subgraph edges: ", temporal_subgraph.edges())
#         print("temporal subgraph nids: ", temporal_subgraph.ndata[dgl.NID])
        

        # The added new edgge will be preserved hence
        root2sub_dict = dict(
            zip(temp2origin.tolist(), temporal_subgraph.nodes().tolist()))
        
        
        # temporal_subgraph.ndata["_orgID"] = g.ndata[dgl.NID][temp2origin]
        # print("temporal subgraph nids: ", temporal_subgraph.ndata[dgl.NID])
        # print("seed node before updates: ", seed_nodes)
        seed_nodes = [root2sub_dict[int(n)] for n in seed_nodes]
        
        # print("updated seed nodes: ", seed_nodes)
        # print("")
        final_subgraph = self.sampler(g=temporal_subgraph, nodes=seed_nodes)
        final_subgraph.remove_self_loop()
        # print("final subgraphs: ", final_subgraph)
        # print("Eids:", final_subgraph.edges())
        # print("Nids:", final_subgraph.nodes())
        # print("ndata: ", final_subgraph.ndata)
        src_nodes, dest_nodes =  final_subgraph.edges()
        src_nodes = src_nodes.unique()
        dest_nodes = dest_nodes.unique()
        # print("s: ", src_nodes)
        # print("d: ", dest_nodes)
        # block = dgl.transforms.to_block(final_subgraph, seed_nodes)
        # block2 = dgl.transforms.to_block(final_subgraph)
        
#         print("block: ", block)
#         print("block ndata: ", block.ndata)
#         print("block srcdata: ", block.srcdata)
#         print("block dstdata: ", block.dstdata)
        
        # print("block2: ", block2)
        # print("block2 ndata: ", block2.ndata)
        # print("block2 srcdata: ", block2.srcdata)
        # print("block2 dstdata: ", block2.dstdata)
        return src_nodes, dest_nodes, final_subgraph #dgl.transforms.to_block(final_subgraph)

        # Temporal Subgraph

    def sample_blocks(self,
                      g,
                      seed_nodes, timestamp,
                      exclude_eids=None):
        blocks = []
        s, d, frontier = self.sampler_frontier(0, g, seed_nodes, timestamp)
        #block = transform.to_block(frontier,seed_nodes)
        block = frontier
        # if self.return_eids:
        #     self.assign_block_eids(block, frontier)
        blocks.append(block)
        # print(blocks)
        # # g.srcnodes, g.srcdata, feature_names
        # print("block src nodes: ", block.srcnodes)
        # print("block src data: ", block.srcdata)
        # # print("node feats prefetch: ", self.prefetch_node_feats)
        return (s, d, blocks)
    def sample_blocks_for_Batch(self,
                      g,
                      seed_nodes, timestamp,
                      exclude_eids=None):
        blocks = []
        s, d, frontier = self.sampler_frontier_for_Batch(0, g, seed_nodes, timestamp)
        #block = transform.to_block(frontier,seed_nodes)
        block = frontier
        # if self.return_eids:
        #     self.assign_block_eids(block, frontier)
        blocks.append(block)
        # print(blocks)
        # # g.srcnodes, g.srcdata, feature_names
        # print("block src nodes: ", block.srcnodes)
        # print("block src data: ", block.srcdata)
        # # print("node feats prefetch: ", self.prefetch_node_feats)
        return (s, d, blocks)


























class MergedSampler(Sampler):

    def __init__(
        self,
        sampler,
        dgl_sampler = None,
        exclude=None,
        reverse_eids=None,
        reverse_etypes=None,
        negative_sampler=None,
        prefetch_labels=None,
        # prefetch_samples = None,
    ):
        super().__init__()
        # Check if the sampler's sample method has an optional third argument.
        # argspec = inspect.getfullargspec(sampler.sample)
        # if len(argspec.args) < 4:  # ['self', 'g', 'indices', 'exclude_eids']
        #     raise TypeError(
        #         "This sampler does not support edge or link prediction; please add an"
        #         "optional third argument for edge IDs to exclude in its sample() method."
        #     )
        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.exclude = exclude
        self.sampler = sampler
        self.dgl_sampler = dgl_sampler
        self.negative_sampler = negative_sampler
        self.prefetch_labels = prefetch_labels or []
        # self.output_device = sampler.output_device

    def sample(self, g, seeds):  # pylint: disable=arguments-differ
        # print("seed shape b: ", seeds)
        part = int(seeds.shape[0]/2)
        # print("part: ", part)   
        seed_edges = seeds[0:part]
        blocks = seeds[part:]

        # root_nodes = [g.edges()[0][seed_edges], g.edges()[1][seed_edges]]
        # time_stamps_old = g.edata['timestamp'][seed_edges]

        block_length = torch.max(blocks).item()+1
        # blocked_batches = [[] for i in range(block_length)]
        blockwise_edge = [[] for i in range(block_length)]

        for i in range(len(seed_edges)):
            blockwise_edge[blocks[i].item()].append(seed_edges[i].item())

        tblocks = []
        rns = []
        tss = []
        npgs = []
        for blk_edgs in blockwise_edge:
            ppg = g.edge_subgraph(blk_edgs, relabel_nodes=False)
            temp = torch.tensor(blk_edgs, dtype=int)
            npg  =  _build_neg_graph(g, temp, self.negative_sampler)
            npgs.append(npg)
            set_edge_lazy_features(ppg, self.prefetch_labels)
            # print("before compact: ")
            # print(npg.ndata)
            # print(npg.edges())
            ppg, npg = compact_graphs([ppg, npg])
            # print("After compact: ")
            # print(npg.ndata)
            # print(npg.edges())
            
            s = g.edges()[0][blk_edgs]
            d = g.edges()[1][blk_edgs]
            t = g.edata['timestamp'][blk_edgs]

            # print("npg: ", npg.edges())
            # print(npg.ndata)
            _, dst = npg.edges()
            neg_dst = npg.ndata[dgl.NID][dst]
            # print("neg_dst: ", neg_dst)
            # print("neg_dst_time: ", ppg.edata['timestamp'])
            # print("ppg: ", ppg.edges())
            # print(ppg.ndata)
            # print(ppg.edata['timestamp'])
          
            # root_nodes = np.concatenate([s.numpy(), d.numpy()])
            # ts =  np.concatenate([t.numpy(),t.numpy()])

            root_nodes = np.concatenate([s.numpy(), d.numpy(), s.numpy(), neg_dst.numpy()])
            # root_nodes = np.concatenate([s.numpy(), d.numpy(), neg_dst.numpy()])

            # print("root nodes: ", root_nodes)
            ts =  np.concatenate([t.numpy(),t.numpy(), t.numpy(), t.numpy()])
            # ts =  np.concatenate([t.numpy(),t.numpy(), t.numpy()])
            # print("ts: ", ts)
            # root_nodes = np.concatenate(g.edges()[0][blk_edgs].numpy(), g.edges()[1][blk_edgs].numpy())
            # ts = np.concatenate(, g.edata['timestamp'][blk_edgs].numpy())
            # print("root nodes: ", root_nodes)
            # print("time_stamps: ", ts)

            self.sampler.sample(root_nodes, ts)
            ret = self.sampler.get_ret()
            mfgs = to_dgl_graph(ret, 1, g.edata['feats'], root_nodes, reverse=False)

            if mfgs[0][0].num_edges() > 0:
                tblocks.append((mfgs[0][0], ppg, npg))
                rns.append(root_nodes)
                tss.append(ts)
        
        if self.dgl_sampler is not None:
            dgl_blks = self.dgl_sampler.sample(g, seeds,npgs = npgs)
            verify(tblocks, dgl_blks, rns, tss)

        return tblocks





# blockwise_edge[block_id] = torch.tensor(blockwise_edge[block_id], dtype=int)
#             bpg_o = g.edge_subgraph(blockwise_edge[block_id], relabel_nodes=False, output_device=self.output_device)
#             set_edge_lazy_features(bpg_o, self.prefetch_labels)
#             nng = self._build_neg_graph(g, blockwise_edge[block_id])

#             # print("bpg_o: ", bpg_o)
#             # print(bpg_o.ndata)
#             # print(bpg_o.edata)

#             bng, nng = compact_graphs([bpg_o, nng])
#             # print("bpg_o: ", bpg_o)
#             # print("bng: ", bng)
#             # print(bpg_o.ndata)
#             # print(bng.ndata)
#             # print(bpg_o.edata)
#             # print(bng.edata)

#             # print(nng.ndata)
#             # print(nng.edata)
            
#             block_pair_graphs.append(bng)
#             block_n_g.append(nng)




class BatchedTemporalEdgePredictionSampler(Sampler):
    """Sampler class that wraps an existing sampler for node classification into another
    one for edge classification or link prediction.

    See also
    --------
    as_edge_predIt finds all the nodes that have zero in-degree and zero out-degree in all the given graphs, and eliminates them from all the graphs.iction_sampler
    """

    def __init__(
        self,
        sampler,
        exclude=None,
        reverse_eids=None,
        reverse_etypes=None,
        negative_sampler=None,
        prefetch_labels=None,
        # prefetch_samples = None,
    ):
        super().__init__()
        # Check if the sampler's sample method has an optional third argument.
        argspec = inspect.getfullargspec(sampler.sample)
        if len(argspec.args) < 4:  # ['self', 'g', 'indices', 'exclude_eids']
            raise TypeError(
                "This sampler does not support edge or link prediction; please add an"
                "optional third argument for edge IDs to exclude in its sample() method."
            )
        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.exclude = exclude
        self.sampler = sampler
        self.negative_sampler = negative_sampler
        self.prefetch_labels = prefetch_labels or []
        self.output_device = sampler.output_device
        # self.prefetch_samples = prefetch_samples

    def _build_neg_graph(self, g, seed_edges):
        neg_srcdst = self.negative_sampler(g, seed_edges)
        # print("neg_srcdst: ", neg_srcdst)
        if not isinstance(neg_srcdst, Mapping):
            assert len(g.canonical_etypes) == 1, (
                "graph has multiple or no edge types; "
                "please return a dict in negative sampler."
            )
            neg_srcdst = {g.canonical_etypes[0]: neg_srcdst}

        dtype = F.dtype(list(neg_srcdst.values())[0][0])
        ctx = context_of(seed_edges) if seed_edges is not None else g.device
        neg_edges = {
            etype: neg_srcdst.get(
                etype,
                (
                    F.copy_to(F.tensor([], dtype), ctx=ctx),
                    F.copy_to(F.tensor([], dtype), ctx=ctx),
                ),
            )
            for etype in g.canonical_etypes
        }
        neg_pair_graph = heterograph(
            neg_edges, {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        )
        return neg_pair_graph
    
    def _build_neg_graph_with_neg_edges(self, g, neg_edge_src, neg_edge_dst, seed_edges):
        # neg_srcdst = self.negative_sampler(g, seed_edges)
        
        neg_srcdst = (torch.tensor(neg_edge_src), torch.tensor(neg_edge_dst))
        
        # print("neg_srcdst: ", neg_srcdst)
        if not isinstance(neg_srcdst, Mapping):
            assert len(g.canonical_etypes) == 1, (
                "graph has multiple or no edge types; "
                "please return a dict in negative sampler."
            )
            neg_srcdst = {g.canonical_etypes[0]: neg_srcdst}

        dtype = F.dtype(list(neg_srcdst.values())[0][0])
        ctx = context_of(seed_edges) if seed_edges is not None else g.device
        neg_edges = {
            etype: neg_srcdst.get(
                etype,
                (
                    F.copy_to(F.tensor([], dtype), ctx=ctx),
                    F.copy_to(F.tensor([], dtype), ctx=ctx),
                ),
            )
            for etype in g.canonical_etypes
        }
        neg_pair_graph = heterograph(
            neg_edges, {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        )
        
        # print("neg_pair_graph: ", neg_pair_graph)
        
        return neg_pair_graph

    def assign_lazy_features(self, result):
        """Assign lazy features for prefetching."""
        pair_graph = result[1]
        set_edge_lazy_features(pair_graph, self.prefetch_labels)
        # In-place updates
        return result

    def sample(self, g, seeds, npgs = None):  # pylint: disable=arguments-differ
        # print("seed shape b: ", seeds)
        part = int(seeds.shape[0]/2)
        # print("part: ", part)   
        seed_edges = seeds[0:part]
        blocks = seeds[part:]

        # if (vb(blocks.tolist())==False):
        #     print("Error dectected")
        # print("seed_edges: ", seed_edges)
        # print("blocks: ", blocks)
        block_length = torch.max(blocks).item()+1
        blocked_batches = [[] for i in range(block_length)]
        blockwise_edge = [[] for i in range(block_length)]
        tss = [[] for i in range(block_length)]
        # print(blocked_batches)

        




        time_stamps_old = g.edata['timestamp'][seed_edges]
        time_stamps = []
        

        if isinstance(seed_edges, Mapping):
            seed_edges = {
                g.to_canonical_etype(k): v for k, v in seed_edges.items()
            }
        exclude = self.exclude

        
        

        #independent execution possible
        for i, edge in enumerate(zip(g.edges()[0][seed_edges], g.edges()[1][seed_edges])):
            # ts = pair_graph.edata['timestamp'][i]
            # print("Edge: ", edge)
            ts = g.edata['timestamp'][seed_edges[i]]
            time_stamps.append(ts)
        
            _, _, subgs = self.sampler.sample_blocks_for_Batch(g,list(edge),timestamp=ts)
            subg = subgs[0]
            # print("positive subg: ",subg)
            # print(subg.edges())
            # print(subg.ndata)
            # print(subg.edata)
            subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
            # print("inserted in block: ", blocks[i].item())
            blocked_batches[blocks[i].item()].append(subg)
            blockwise_edge[blocks[i].item()].append(seed_edges[i])
            tss[blocks[i].item()].append(ts)



        # for i in range(len(blockwise_edge)):
        #     if len(blockwise_edge[i])==0:
        #         print("Error Detected here 2")


        #independent parallel execution possible
        block_pair_graphs = []
        block_n_g = []
        for block_id in range(len(blockwise_edge)):
            blockwise_edge[block_id] = torch.tensor(blockwise_edge[block_id], dtype=int)
            bpg_o = g.edge_subgraph(blockwise_edge[block_id], relabel_nodes=False, output_device=self.output_device)
            set_edge_lazy_features(bpg_o, self.prefetch_labels)
            if npgs is not None:
                nng = npgs[block_id]
                # print("inside old: ", nng.ndata)
                # print(nng.edges())
            else:
                nng = self._build_neg_graph(g, blockwise_edge[block_id])

            # print("bpg_o: ", bpg_o)
            # print(bpg_o.ndata)
            # print(bpg_o.edata)

            bng, nng = compact_graphs([bpg_o, nng])
            # print(nng.ndata)
            # print(nng.edges())
            # print("bpg_o: ", bpg_o)
            # print("bng: ", bng)
            # print(bpg_o.ndata)
            # print(bng.ndata)
            # print(bpg_o.edata)
            # print(bng.edata)
            # print("npg after compact in old: ")
            # print(nng.edges())
            # print(nng.ndata)
            # print(nng.edata)
            
            block_pair_graphs.append(bng)
            block_n_g.append(nng)

        #independent parallel execution possible
        # print("blockwise_edge: ", blockwise_edge)
        gi = 0

        for block_id in range(len(block_n_g)):
            bi = 0
            ng = block_n_g[block_id]
            src, dst  = ng.edges()
            # print("befor sampling: ")
            # print(ng.ndata)
            # print(ng.edges())
        
            b_neg_list = []
            for i, src_node in enumerate(src):
                b_neg_list.append([ng.ndata[dgl.NID][src_node], ng.ndata[dgl.NID][dst[i]]])
            for i, neg_edge in enumerate(b_neg_list):
                ts = time_stamps[gi]
                gi += 1
                
                ts = tss[block_id][bi]
                bi += 1
                # new_neg_edge_src.append(neg_edge[0])
                # new_neg_edge_dst.append(neg_edge[1])
                _, _, subgs = self.sampler.sample_blocks_for_Batch(g,neg_edge,timestamp=ts)
                subg = subgs[0]
                # print("Block: ", block_id)
                # print("Neg edge: ", neg_edge)
                # print("Negative subg: ___________________________________________________________________",subg)

                # print(subg.edges())
                # print(subg.ndata)
                # print(subg.edata)
                subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())



                blocked_batches[block_id].append(subg)

        batched_t_blocks = []        
        # for t_block in blocked_batches:
        #     batched_t_blocks.append(dgl.batch(t_block))
                                     
        for i in range(len(blocked_batches)):
            t_block = blocked_batches[i]

            # if(len(t_block)==0):
            #     print("Empty block: ", i, "\n..............", blockwise_edge[i])

            bng = block_pair_graphs[i]
            nng = block_n_g[i]
            # bng, nng = compact_graphs([bpg_o, nng])
            # print("Mini NG new version: ", printEdges(nng))
            batched_t_blocks.append((dgl.batch(t_block), bng, nng))
            


        a, b, c  = batched_t_blocks[0]
        # print("Samplera: ", a) 
        # print("Samplerb: ", b) 
        # print("Samplerc: ", c) 
        # print(a.edges())
        # print(a.ndata)
        # print(a.edata)

        return batched_t_blocks
        # if self.negative_sampler is None:
        #     tmp =  self.assign_lazy_features((input_nodes, pair_graph_v2, batched_t_blocks))
        # else:
        #     tmp = self.assign_lazy_features((input_nodes, pair_graph_v2, neg_graph_v2, batched_t_blocks))
        
"""





        pair_graph = g.edge_subgraph(
            seed_edges, relabel_nodes=False, output_device=self.output_device
        )
        
        pair_graph_v2 = g.edge_subgraph(
            seed_edges, relabel_nodes=False, output_device=self.output_device
        )
        positive_seed_nodes = pair_graph.ndata[NID]
        
        eids = pair_graph.edata[EID]
        # print("Pair graph: ", pair_graph)

        if self.negative_sampler is not None:
            # print("Do Noting...........................")
            neg_graph = self._build_neg_graph(g, seed_edges)
            # print("neg_graph ndata: ", neg_graph.ndata)
            # print("neg_graph edges: ", neg_graph.edges())
            pair_graph, neg_graph = compact_graphs([pair_graph, neg_graph])
            # print("compact neg_graph ndata: ", neg_graph.ndata)
            
        else:
            pair_graph = compact_graphs(pair_graph)
            # print("Pair_graph edges: ", pair_graph.edges())
        # print("compact pair graph ndata: ",pair_graph.ndata)
        # print("edges: ", pair_graph.edges())

        pair_graph.edata[EID] = eids
        seed_nodes = pair_graph.ndata[NID]
        
        
        batch_graphs = []
        nodes_id = []
        timestamps = []
        exclude_eids = find_exclude_eids(
            g,
            seed_edges,
            exclude,
            self.reverse_eids,
            self.reverse_etypes,
            self.output_device,
        )




        last_accessed_dict = {}
        access_dict = {}
        t_access_dict = {}
        t_blocks = []
        
        idx = 0
        blockwise_edge = []
        bis = []
        
        for i, edge in enumerate(zip(g.edges()[0][seed_edges], g.edges()[1][seed_edges])):
            # ts = pair_graph.edata['timestamp'][i]
            # print("Edge: ", edge)
            ts = pair_graph.edata['timestamp'][i]
            timestamps.append(ts)
            
            _, _, subgs = self.sampler.sample_blocks_for_Batch(g,list(edge),timestamp=ts)
            subg = subgs[0]
            subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
            nodes_id.append(subg.srcdata[dgl.NID])



            if()


            batch_graphs.append(subg)
            
            block_id = InserInBlock(subg, last_accessed_dict, t_blocks, access_dict, t_access_dict, ts)
            edge_idx = seed_edges[idx].item()
            
            
            if len(blockwise_edge) <= block_id:
                blockwise_edge.append([])
                bis.append([])
            blockwise_edge[block_id].append(edge_idx)
            bis[block_id].append(idx)
            idx += 1
            

        
        block_pair_graphs = []
        block_n_g = []
        p_edges_in_order = []
        for block_id in range(len(blockwise_edge)):
            blockwise_edge[block_id] = torch.tensor(blockwise_edge[block_id])
            bpg_o = g.edge_subgraph(blockwise_edge[block_id], relabel_nodes=False, output_device=self.output_device)
            set_edge_lazy_features(bpg_o, self.prefetch_labels)
            nng = self._build_neg_graph(g, blockwise_edge[block_id])
            bng, nng = compact_graphs([bpg_o, nng])
            
            block_pair_graphs.append(bng)
            block_n_g.append(nng)


        
        neg_edges = get_neg_edges_and_ts(t_access_dict, t_blocks, positive_seed_nodes, g)
        timestamps = torch.tensor(timestamps).repeat_interleave(self.negative_sampler.k)
        # for i, neg_edge in enumerate(zip(neg_srcdst_raw[0].tolist(), neg_srcdst_raw[1].tolist())):
        neg_list = []
        gi = 0
        new_neg_edge_src = []
        new_neg_edge_dst = []



        for block_id in range(len(block_n_g)):
            ng = block_n_g[block_id]
            src, dst  = ng.edges()
            b_neg_list = []
            for i, src_node in enumerate(src):
                b_neg_list.append([ng.ndata[dgl.NID][src_node], ng.ndata[dgl.NID][dst[i]]])
            for i, neg_edge in enumerate(b_neg_list):
                ts = timestamps[gi]
                gi += 1
                new_neg_edge_src.append(neg_edge[0])
                new_neg_edge_dst.append(neg_edge[1])
                _, _, subgs = self.sampler.sample_blocks_for_Batch(g,neg_edge,timestamp=ts)
                subg = subgs[0]
                subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
                t_blocks[block_id].append(subg)
                
        
        src, dst = neg_graph.edges()
        for i, src_node in enumerate(src):
            neg_list.append([neg_graph.ndata[dgl.NID][src_node], neg_graph.ndata[dgl.NID][dst[i]]])
        for i, neg_edge in enumerate(neg_list):
            ts = timestamps[i]
            # print(ts)
            # print("neg_edge: ", neg_edge)
            _, _, subgs = self.sampler.sample_blocks_for_Batch(g,
                                                    neg_edge,
                                                    timestamp=ts)
            subg = subgs[0]
            # print("original neg edge: ", neg_edge)
            # print("subg ndata: ", subg.edges())
            # print("subg ndata: ", subg.ndata)
            # t_dict = {}
            # t_dict["_N"] = ts.repeat(subg.num_nodes(ntype="_N"))
            # subg.ndata['timestamp'] = t_dict#ts.repeat(subg.num_nodes(ntype="_N"))
            subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
            # subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
            batch_graphs.append(subg)
        blocks = [dgl.batch(batch_graphs)]
        input_nodes = torch.cat(nodes_id)
        # return input_nodes, pair_graph, neg_pair_graph, blocks
        
        
        neg_edge_src = []
        neg_edge_dst = []
        
        block_wise_neg_src = []
        block_wise_neg_dst = []
        
        for block_id in range(len(t_blocks)):
            
            block_wise_neg_src.append([])
            block_wise_neg_dst.append([])
            neg_edg_block = neg_edges[block_id]
            
            # if block_id == 0:
            #     print("Neg edges: ", neg_edg_block)

            for neg_edge_tuple in neg_edg_block:
                neg_edg, ts =  neg_edge_tuple
                # print()
                neg_edge_src.append(neg_edg[0])
                neg_edge_dst.append(neg_edg[1])
                
                block_wise_neg_src[block_id].append(neg_edg[0])
                block_wise_neg_dst[block_id].append(neg_edg[1])
                
                ts = torch.tensor(ts, dtype=torch.float64)
                _, _, subgs = self.sampler.sample_blocks_for_Batch(g,
                                                    neg_edg,
                                                    timestamp=ts)
                subg = subgs[0]
                # print("Negative subg: ", subg)
                # print("Negative subg edges: ", subg.edges())
                # print("Negative subg ndata: ", subg.ndata)
                subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
                # t_blocks[block_id].append(subg) #Reactivate
        
       #*********************************************************************************** 
        batched_t_blocks = []        
        # for t_block in t_blocks:
        #     batched_t_blocks.append(dgl.batch(t_block))
                                     
        for i in range(len(t_blocks)):
            t_block = t_blocks[i]
            bng = block_pair_graphs[i]
            nng = block_n_g[i]
            # bng, nng = compact_graphs([bpg_o, nng])
            # print("Mini NG new version: ", printEdges(nng))
            batched_t_blocks.append((dgl.batch(t_block), bng, nng))
            
        
        
        
        
        # neg_graph_v2 = self._build_neg_graph_with_neg_edges(g, neg_edge_src, neg_edge_dst, seed_edges)
        
        neg_graph_v2 = self._build_neg_graph_with_neg_edges(g, new_neg_edge_src, new_neg_edge_dst, seed_edges)
        pair_graph_v2, neg_graph_v2 = compact_graphs([pair_graph_v2, neg_graph_v2])
        
        if self.negative_sampler is None:
            tmp =  self.assign_lazy_features((input_nodes, pair_graph_v2, batched_t_blocks))
        else:
            tmp = self.assign_lazy_features((input_nodes, pair_graph_v2, neg_graph_v2, batched_t_blocks))
        
        return tmp

#         if self.negative_sampler is None:
#             return self.assign_lazy_features((input_nodes, pair_graph, blocks))
#         else:
#             return self.assign_lazy_features((input_nodes, pair_graph, neg_graph, blocks))


"""

# def getNegativeSrcDstCandidates(graph, exclude_nids):
#     src, dst  = graph.edges()
#     src = src.unique().tolist()
#     dst = dst.unique().tolist()
    
#     src = graph.ndata["_ID"][src].unique()
#     dst = graph.ndata["_ID"][dst].unique()
    
#     filtered_srcs = []
#     filtered_dsts = []
#     for nid in src:
#         if nid.item() not in exclude_nids:
#             filtered_srcs.append(nid.item())
#     for nid in dst:
#         if nid.item() not in exclude_nids:
#             filtered_dsts.append(nid.item())
#     return filtered_srcs, filtered_dsts
















def to_dgl_graph(ret, hist, edge_feats, root_nodes, reverse=False, cuda=True, id_bak = False):
    mfgs =  list()
    id_name = dgl.NID
    if id_bak:
        id_name = "ID"
    for r in ret:
        # if use_sk and sk:
        #     print(r.col())
        #     print(r.row())
        #     print(r.nodes())

        #     k = input("utils ln 49: type stop to avoid blocking next time")
        #     if k  == 'stop':
        #         sk = False
        if not reverse:
            g = dgl.graph((r.col(), r.row()), num_nodes=len(r.nodes()))
        else:
            g = dgl.graph((r.row(), r.col()), num_nodes=len(r.nodes()))
        g.ndata[id_name] = torch.from_numpy(r.nodes())
        g.ndata['timestamp'] = torch.from_numpy(r.ts())
        # print("r.eid(): ", r.eid())
        ID = torch.from_numpy(r.eid())
        dt = torch.from_numpy(r.dts())[r.dim_out():]
        if edge_feats is not None:
            feats =  edge_feats[ID]
        # feat = embedding[feat_id]
        # print(root_nodes)
        self_loop =  False
        if self_loop:
    
            g.add_edges(torch.arange(0,len(root_nodes)), torch.arange(0,len(root_nodes)))
            
            g.edata[id_name] = torch.arange(0, g.number_of_edges(), dtype=torch.int64)


            # print(g.number_of_edges())
            # print(feats.shape)
            
            
            # print(zeros.shape)
            if edge_feats is not None:
                zeros  = torch.zeros(len(root_nodes), feats.shape[1])
                g.edata['feats'] = torch.cat((feats, zeros), dim=0)


            g.edata['ID'] = torch.cat((ID, torch.zeros(len(root_nodes))), dim=0)
            g.edata['timestamp'] = torch.cat((dt, torch.zeros(len(root_nodes))), dim=0)
        else:
            g.edata['ID'] = ID#torch.cat((ID, torch.zeros(len(root_nodes))), dim=0)
            g.edata['timestamp'] = dt#torch.cat((dt, torch.zeros(len(root_nodes))), dim=0)
            if edge_feats is not None:
                g.edata['feats'] = feats#torch.cat((feats, zeros), dim=0)
        

        if cuda:
            mfgs.append(g.to('cuda:0'))
        else:
            mfgs.append(g)
    mfgs = list(map(list, zip(*[iter(mfgs)] * hist)))
    mfgs.reverse()
    return mfgs



def verify(tblocks, dgl_blks, rn, ts):
    if len(tblocks) != len(dgl_blks):
            print("len(tblocks): ", len(tblocks))
            print("len(dgl_blocks): ", len(dgl_blks))
    else:
            # print("batch: ", batch_cnt)
            # print(tblocks[0])
            a, b, c = dgl_blks[0]
            # printG(tblocks[0])
            # printG(a)
            for i in range(len(tblocks)):
                tblk, p, n = tblocks[i]
                cput = tblk.to('cpu')
                a, b, c  =  dgl_blks[i]
                
                
                are_isomorphic = check_isomorphism(cput, a)
                # are_isomorphic = check_isomorphism(g1, g2)
                print("RN and TS: ", rn[i], ts[i])
                if are_isomorphic:
                    print("The graphs are isomorphic.")
                else:
                    print("The graphs are not isomorphic.")
                    printG(tblk)
                    print("________________")
                    printG(a)
                    print("++++++++++++++++")
                    input("vs ln 173")


def printG(g, ndata =True, edges = True, edata = False):
    print("Graph: ", g)
    if ndata:
        print("ndata: ", g.ndata)
    if edges:
        print("edges: ", g.edges())
    if edata:
        print("edata: ", g.edata)
    else:
        print("edata: ", g.edata['timestamp'])



def _build_neg_graph(g, seed_edges, negative_sampler):
    neg_srcdst = negative_sampler(g, seed_edges)
    t1, t2  = neg_srcdst
    mask = t1 == t2
    t2[mask] += 1
    neg_srcdst = (t1, t2)

    # print("neg_srcdst: ", neg_srcdst)
    if not isinstance(neg_srcdst, Mapping):
        assert len(g.canonical_etypes) == 1, (
            "graph has multiple or no edge types; "
            "please return a dict in negative sampler."
        )
        neg_srcdst = {g.canonical_etypes[0]: neg_srcdst}

    dtype = F.dtype(list(neg_srcdst.values())[0][0])
    ctx = context_of(seed_edges) if seed_edges is not None else g.device
    neg_edges = {
        etype: neg_srcdst.get(
            etype,
            (
                F.copy_to(F.tensor([], dtype), ctx=ctx),
                F.copy_to(F.tensor([], dtype), ctx=ctx),
            ),
        )
        for etype in g.canonical_etypes
    }
    neg_pair_graph = heterograph(
        neg_edges, {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    )
    return neg_pair_graph