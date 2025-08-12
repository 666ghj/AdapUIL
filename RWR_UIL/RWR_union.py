import networkx as nx
import numpy as np
import torch.nn.functional as F
import torch
from RWR_UIL.RWR_DeepWalk import RWR_DeepWalk


def RWR_union(src_edges, tar_add5000, gt_tar_add5000, alpha, emb_size,
              length_walk, num_walks, window_size, num_iters,union):
    union_edges = np.vstack((src_edges, tar_add5000))  # 将两个Graph垂直方向拼接
    all_nodes = union_edges[:, :].flatten()
    all_nodes = np.unique(all_nodes)
    src_node_num = len(np.unique(src_edges[:, :].flatten()))
    tar_node_num = len(np.unique(tar_add5000[:, :].flatten()))

    print('nodes_num')
    node_num = len(all_nodes)
    print(node_num)

    np.random.shuffle(union_edges)
    train_union_edges = union_edges[:, :]

    edges = train_union_edges
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(edges)

    # 嵌入大小，序列长度，游走轮数
    deepwalk = RWR_DeepWalk(node_num, edges, G, alpha, emb_size, length_walk, num_walks, window_size, num_iters,
                            gt_tar_add5000,union)
    w2v = deepwalk.train(workers=4, is_loadmodel=False, is_loaddata=False, node_type="union")

    src_vector_source = []
    tar_vector_source = []
    # 本来是想直接轮询取出Embedding值，后来发现需要排序节点，要不然concate或者相加的时候对应不起来，所以这两行代码注释了
    # for word in list(w2v.wv.index_to_key):
    #     vector_source.append(w2v.wv[word])
    for i in range(src_node_num):
        src_vector_source.append(w2v.wv[str(i)])
    src_vector_source = np.array(src_vector_source)  # list转numpy的array

    src_vector_source = torch.FloatTensor(src_vector_source)  # numpy的array转tensor
    src_vector_source = F.normalize(src_vector_source, dim=1) # 每行都除以该行下所有元素平方和的开方，标准化
    src_vector_source = src_vector_source.numpy()  # 再转回numpy的array类型

    for i in range(tar_node_num):
        tar_vector_source.append(w2v.wv[str(i+4096)])
    tar_vector_source = np.array(tar_vector_source)

    tar_vector_source = torch.FloatTensor(tar_vector_source)
    tar_vector_source = F.normalize(tar_vector_source, dim=1)
    tar_vector_source = tar_vector_source.numpy()

    return src_vector_source,tar_vector_source
