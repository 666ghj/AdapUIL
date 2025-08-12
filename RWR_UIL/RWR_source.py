import networkx as nx
import numpy as np
import math
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from RWR_UIL.RWR_DeepWalk import RWR_DeepWalk
from source.deepwalk import DeepWalk


def RWR_source(source_edges, groundtruth, src_neg_edges, alpha, emb_size, length_walk, num_walks, window_size,
               num_iters):
    all_nodes = source_edges[:, :].flatten()
    all_nodes = np.unique(all_nodes)
    node_num = len(all_nodes)
    print("----------------------当前网络source，节点个数：", node_num)

    np.random.shuffle(source_edges)

    online_len = source_edges.shape[0]
    train_size = math.floor(0.8 * online_len)
    val_size = math.floor(0.1 * online_len)
    # test_size = math.floor(0.1 * online_len)
    train_online_edges = source_edges[:, :]
    # val_online_edges = online_edges[train_size:val_size+train_size,:]
    test_size_edges = source_edges[val_size + train_size:, :]

    edges = train_online_edges
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(edges)
    # deepwalk = DeepWalk(G, emb_size, length_walk,num_walks, window_size, num_iters)
    # 嵌入大小，序列长度，游走轮数
    # deepwalk = Urp_DeepWalk(node_num,edges,G, emb_size=128, length_walk=50,num_walks=10, window_size=10, num_iters=2)
    deepwalk = RWR_DeepWalk(node_num, edges, G, alpha, emb_size, length_walk, num_walks, window_size, num_iters)
    w2v = deepwalk.train(workers=4, is_loadmodel=False, is_loaddata=False, node_type="source")

    pos_test = test_size_edges
    neg_test = src_neg_edges

    y_true = [True] * pos_test.shape[0] + [False] * neg_test.shape[0]
    X = np.vstack([pos_test, neg_test])
    '''
    print('Testing...')
    y_score = []
    for u, v in X:
        # if u==3885 or v==3885:
        #     print(u,v)
        y_score.append(w2v.wv.similarity(str(u), str(v)))
    '''
    vector_source = []

    for i in range(node_num):
        vector_source.append(w2v.wv[str(i)])
    vector_source = np.array(vector_source)
    vector_source = torch.FloatTensor(vector_source)
    vector_source = F.normalize(vector_source, dim=1)
    vector_source = vector_source.numpy()

    s_anchor = []
    for i in groundtruth:
        s_anchor.append(i[0])

    anchor_list = []
    for i in range(node_num):
        if i in s_anchor:
            anchor_list.append(1)
        else:
            anchor_list.append(2)

    # viz = visdom.Visdom()
    # down_embedding = TSNE(n_components=2).fit_transform(vector_source)
    # print(down_embedding)
    # x = []
    # y = []
    # for i in down_embedding:
    #     x.append(i[0])
    #     y.append(1)
    # x = np.array(x)
    # y = np.array(anchor_list)
    # color = np.array([[47,16,244],[237,26,26]])
    # win  = viz.scatter(
    #     X = down_embedding,
    #     Y = y,
    #     opts=dict(
    #         markersize = 10,
    #         markersymbol = 'cross-thin-open',
    #         markercolor = color,
    #         legend = ['1','2'],
    #     ),
    # )
    '''
    auc_test = roc_auc_score(y_true, y_score)
    print('Tencent, test AUC:', auc_test)
    '''
    return vector_source
