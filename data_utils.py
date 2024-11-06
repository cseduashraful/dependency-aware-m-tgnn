import os
import dgl
import ssl
from six.moves import urllib
import numpy as np
import pandas as pd
import torch
from dgl.data import GDELTDataset
from tqdm import tqdm
import itertools

def convert2tcsr(g):
    num_nodes = g.num_nodes()
    ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
    ext_full_indices = [[i] for i in range(num_nodes)]
    ext_full_ts = [[0] for _ in range(num_nodes)]
    ext_full_eid = [[0] for _ in range(num_nodes)]  #need to recheck

    srcs, dsts, eids = g.edges(form='all')
    tss = g.edata['timestamp']
    cont  = 1
    for u, v, idx in zip(srcs, dsts, eids):
        # print(f"Edge from {u.item()} to {v.item()} with Edge ID {idx.item()} at time {tss[idx.item()].item()}")
        src = v
        dst = u

        if cont == 0:
            cont = input()
        ext_full_indices[src].append(dst)
        ext_full_ts[src].append(tss[idx.item()].item())
        ext_full_eid[src].append(idx)
    
    for i in tqdm(range(num_nodes)):
        ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])
    ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
    ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
    ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

    print('Sorting...')
    def tsort(i, indptr, indices, t, eid):
        beg = indptr[i]
        end = indptr[i + 1]
        sidx = np.argsort(t[beg:end])
        indices[beg:end] = indices[beg:end][sidx]
        t[beg:end] = t[beg:end][sidx]
        eid[beg:end] = eid[beg:end][sidx]

    for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
        # tsort(i, int_train_indptr, int_train_indices, int_train_ts, int_train_eid)
        # tsort(i, int_full_indptr, int_full_indices, int_full_ts, int_full_eid)
        tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

    print(ext_full_indptr)
    print(ext_full_indices)
    print(ext_full_ts)
    print(ext_full_eid)

    return {"indptr": ext_full_indptr, "indices": ext_full_indices, "ts": ext_full_ts, "eid": ext_full_eid}




    # for idx, row in tqdm(g.edges(), total=len(g.num_edges())):

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)



def reindex(df, bipartite=False):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def run(data_name, bipartite=False):
    PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    df, feat = preprocess(PATH)
    new_df = reindex(df, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)

def TemporalDataset(dataset, force_reload = True, bipartite =  False):
    if(dataset == "gedelt"):
        return loadGdeltDataset()
    if force_reload or not os.path.exists('./data/{}.bin'.format(dataset)):
        if not os.path.exists('./data/{}.csv'.format(dataset)):
            if not os.path.exists('./data'):
                os.mkdir('./data')

            url = 'https://snap.stanford.edu/jodie/{}.csv'.format(dataset)
            print("Start Downloading File....")
            context = ssl._create_unverified_context()
            data = urllib.request.urlopen(url, context=context)
            with open("./data/{}.csv".format(dataset), "wb") as handle:
                handle.write(data.read())

        print("Start Process Data ...")
        run(dataset, bipartite = bipartite)
        raw_connection = pd.read_csv('./data/ml_{}.csv'.format(dataset))
        raw_feature = np.load('./data/ml_{}.npy'.format(dataset))
        # -1 for re-index the node
        src = raw_connection['u'].to_numpy()-1
        dst = raw_connection['i'].to_numpy()-1
        # Create directed graph
        g = dgl.graph((src, dst))
        g.edata['timestamp'] = torch.from_numpy(
            raw_connection['ts'].to_numpy())
        g.edata['label'] = torch.from_numpy(raw_connection['label'].to_numpy())
        g.edata['feats'] = torch.from_numpy(raw_feature[1:, :]).float()
        g.ndata[dgl.NID] = g.nodes()
        dgl.save_graphs('./data/{}.bin'.format(dataset), [g])
    else:
        print("Data exists,  directly loaded.")
        gs, _ = dgl.load_graphs('./data/{}.bin'.format(dataset))
        g = gs[0]
    return g


def loadGdeltDataset():
    folder_path = '/project/pi_mserafini_umass_edu/ashraful/tgnn/TGN/tgl/DATA/GDELT'
    # dgl.data.dgl_dataset.DGLBuiltinDataset
    # train_data = GDELTDataset()
    # valid_data = GDELTDataset(mode='valid')
    # test_data = GDELTDataset(mode='test')
    # print("loading done")
    # print(train_data[0].edata)
    g, df = load_graph(folder_path)

    print("Graph is loaded")
    print("g\n",g)
    print("df\n",df)
    ef = load_feat(folder_path)
    print("loading done")
    
    print("edge feature:\n", ef)
    return g


def load_feat(d):
    # node_feats = None
    # if os.path.exists('{}/node_features.pt'.format(d)):
    #     node_feats = torch.load('DATA/{}/node_features.pt'.format(d))
    #     if node_feats.dtype == torch.bool:
    #         node_feats = node_feats.type(torch.float32)
    edge_feats = None
    if os.path.exists('{}/edge_features.pt'.format(d)):
        edge_feats = torch.load('{}/edge_features.pt'.format(d))
        if edge_feats.dtype == torch.bool:
            edge_feats = edge_feats.type(torch.float32)
    # if rand_de > 0:
    #     if d == 'LASTFM':
    #         edge_feats = torch.randn(1293103, rand_de)
    #     elif d == 'MOOC':
    #         edge_feats = torch.randn(411749, rand_de)
    # if rand_dn > 0:
    #     if d == 'LASTFM':
    #         node_feats = torch.randn(1980, rand_dn)
    #     elif d == 'MOOC':
    #         edge_feats = torch.randn(7144, rand_dn)
    return edge_feats

def load_graph(d):
    df = pd.read_csv('{}/edges.csv'.format(d))
    g = np.load('{}/ext_full.npz'.format(d))
    return g, df