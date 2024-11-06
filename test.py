import dgl
import torch

class FullyVectorizedNegativeSampler:
    def __init__(self, k, destination_nodes):
        self.k = k
        self.destination_nodes = torch.tensor(destination_nodes)

    def __call__(self, g, eids):
        src, dst = g.find_edges(eids)
        positive_pairs = set(zip(src.tolist(), dst.tolist()))

        # Expand src to have k repeated entries for each source node
        src_repeated = src.repeat_interleave(self.k)

        # Sample k negative destinations for each source node
        sampled_destinations = self.destination_nodes[torch.randint(0, len(self.destination_nodes), (len(src_repeated),))]

        # Filter out positive pairs using a set for fast lookup
        mask = torch.tensor(
            [(u.item(), v.item()) not in positive_pairs for u, v in zip(src_repeated, sampled_destinations)]
        )

        # Keep only valid negative samples
        negative_src = src_repeated[mask]
        negative_dst = sampled_destinations[mask]

        # If we have more samples than needed, truncate the excess
        if len(negative_src) > len(src) * self.k:
            negative_src = negative_src[:len(src) * self.k]
            negative_dst = negative_dst[:len(src) * self.k]

        return negative_src, negative_dst

# Create a simple graph
u = torch.tensor([0, 1, 2, 3])
v = torch.tensor([4, 5, 5, 6])
g = dgl.graph((u, v))

# Define the number of negative samples per positive edge
num_negative_samples = 1

# Get the unique destination nodes from the original graph
destination_nodes = v.unique().tolist()

# Create the fully vectorized negative sampler
sampler = FullyVectorizedNegativeSampler(k=num_negative_samples, destination_nodes=destination_nodes)

# Use all edges of the graph to create positive edges
positive_edges = torch.arange(g.num_edges())

# Sample negative edges
negative_src, negative_dst = sampler(g, positive_edges)

print("Sampled negative edges:")
print("Source Nodes:", negative_src)
print("Destination Nodes:", negative_dst)
