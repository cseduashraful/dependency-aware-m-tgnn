import dgl
import networkx as nx
from networkx.algorithms import isomorphism

def getSortedEdgePair(g):
    src_indices, dst_indices = g.edges()
    node_ids = g.ndata['_ID']
    src_ids = node_ids[src_indices]
    dst_ids = node_ids[dst_indices]
    edge_list = list(zip(src_ids.tolist(), dst_ids.tolist(), g.edata['timestamp'].tolist()))
    sorted_edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))
    return sorted_edge_list


def check_isomorphism(g1, g2):
    # Convert DGL graphs to NetworkX graphs

    el1 = getSortedEdgePair(g1)
    el2 = getSortedEdgePair(g2)
    if el1 == el2:
        print("The two edge lists are the same.")
    else:
        print("The two edge lists are different.")
        print(el1)
        print("_-----_-----")
        print(el2)
        input("Type something to continue...")   
    nx_g1 = g1.to_networkx(node_attrs=['_ID'], edge_attrs=[])
    nx_g2 = g2.to_networkx(node_attrs=['_ID'], edge_attrs=[])

    # Define a node comparison function based on the '_ID' label
    def node_match(n1, n2):
        return n1['_ID'] == n2['_ID']

    # Check for isomorphism using NetworkX
    gm = isomorphism.GraphMatcher(nx_g1, nx_g2, node_match=node_match)
    
    return gm.is_isomorphic()

# Example usage:
# g1 and g2 are your DGL graphs

