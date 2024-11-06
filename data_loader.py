import torch
import dgl
from dgl.dataloading import BlockSampler
from functools import partial
# from dgl.dataloading import EdgeCollator




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
        super(TemporalSampler, self).__init__(1, False)
        if sampler_type == 'topk':
            self.sampler = partial(
                dgl.sampling.select_topk, k=k, weight='timestamp')
        elif sampler_type == 'uniform':
            self.sampler = partial(dgl.sampling.sample_neighbors, fanout=k)
        else:
            raise DGLError(
                "Sampler string invalid please use \'topk\' or \'uniform\'")

    def sampler_frontier(self,
                         block_id,
                         g,
                         seed_nodes,
                         timestamp):
        full_neighbor_subgraph = dgl.in_subgraph(g, seed_nodes)
        full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,
                                               seed_nodes, seed_nodes)

        temporal_edge_mask = (full_neighbor_subgraph.edata['timestamp'] < timestamp) + (
            full_neighbor_subgraph.edata['timestamp'] <= 0)
        temporal_subgraph = dgl.edge_subgraph(
            full_neighbor_subgraph, temporal_edge_mask)

        # Map preserve ID
        temp2origin = temporal_subgraph.ndata[dgl.NID]

        # The added new edgge will be preserved hence
        root2sub_dict = dict(
            zip(temp2origin.tolist(), temporal_subgraph.nodes().tolist()))
        temporal_subgraph.ndata[dgl.NID] = g.ndata[dgl.NID][temp2origin]
        seed_nodes = [root2sub_dict[int(n)] for n in seed_nodes]
        final_subgraph = self.sampler(g=temporal_subgraph, nodes=seed_nodes)
        final_subgraph.remove_self_loop()
        return final_subgraph

        # Temporal Subgraph
    def sample_blocks(self,
                      g,
                      seed_nodes,
                      timestamp):
        blocks = []
        frontier = self.sampler_frontier(0, g, seed_nodes, timestamp)
        #block = transform.to_block(frontier,seed_nodes)
        block = frontier
        if self.return_eids:
            self.assign_block_eids(block, frontier)
        blocks.append(block)
        return blocks
    
