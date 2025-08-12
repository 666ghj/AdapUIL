'''
1 数据集更换
2 网络结构优化
'''
import time

import numpy as np
import math

from source.NetWork_Opt import netWork_Enhance, del_leaf_nodes


def RWR_get_stg():
    # 豆瓣数据集
    gt = np.genfromtxt("dataspace/douban/groundtruth", dtype=np.int32)  # 1118*2  锚节点对
    get_src = np.genfromtxt("dataspace/douban/online.edgelist", dtype=np.int32)  # 3906  source节点对
    get_tar = np.genfromtxt("dataspace/douban/offline.edgelist", dtype=np.int32)  # 1118   target节点对
    src_neg_edges = np.genfromtxt("dataspace/douban/sourece_neg_douban.txt", dtype=np.int32)
    tar_neg_edges = np.genfromtxt("dataspace/douban/target_neg_douban.txt", dtype=np.int32)


    # Facebook和twitter数据
    # get_src = np.genfromtxt("dataspace/Facebook/twitter.edges",dtype=np.int32) # 3493
    # get_tar = np.genfromtxt("dataspace/Facebook/facebook.edges",dtype=np.int32) # 1792
    # gt = np.genfromtxt("dataspace/Facebook/fb_tw.ground",dtype=np.int32)
    # src_neg_edges = np.genfromtxt("dataspace/Facebook/sourece_neg_tw.txt",dtype=np.int32)
    # tar_neg_edges = np.genfromtxt("dataspace/Facebook/target_neg_fb.txt", dtype=np.int32)

    # 下面三行为两网融合所设计，在文件RWR_union.py里还有个5000是写死的
    gt_tar_add5000 = gt.copy()
    gt_tar_add5000 = gt_tar_add5000[:math.floor(0.8 * len(gt_tar_add5000))]  # 取训练集，排除验证集和测试集
    gt_tar_add5000[:, 1] += 4096  # 增加第二列target的每个元素5000    4096
    tar_add5000 = get_tar + 4096


    # Foursquare_Twitter数据集
    # get_src = np.genfromtxt("../dataspace/fourq.number",dtype=np.int32)   # 5313
    # get_tar = np.genfromtxt("../dataspace/tw.number",dtype=np.int32)     # 5120
    # gt = np.genfromtxt("../dataspace/fq_tw_ground.txt",dtype=np.int32)
    # src_neg_edges = np.genfromtxt("../dataspace/foursquare_source_neg.txt", dtype=np.int32)
    # tar_neg_edges = np.genfromtxt("../dataspace/twitter_target_neg.txt", dtype=np.int32)

    # get_src = np.genfromtxt("../dataspace/flickr.edgelist",dtype=np.int32)
    # get_tar = np.genfromtxt("../dataspace/lastfm.edgelist",dtype=np.int32)
    # gt = np.genfromtxt("../dataspace/flickr_lastfm_groundtruth",dtype=np.int32)
    # src_neg_edges = np.genfromtxt("../dataspace/flickr_source_neg.txt", dtype=np.int32)
    # tar_neg_edges = np.genfromtxt("../dataspace/lastfm_target_neg.txt", dtype=np.int32)

    # get_src = np.genfromtxt("../dataspace/dblp1.edges",dtype=np.int32)
    # get_tar = np.genfromtxt("../dataspace/dblp2.edges",dtype=np.int32)
    # groundtruth = np.genfromtxt("../dataspace/dblp.alignment",dtype=np.int32)

    # 网络结构优化
    get_src_nodes = np.unique(get_src[:, :].flatten())  # source中所有存在的节点集合
    get_tar_nodes = np.unique(get_tar[:, :].flatten())  # target中所有存在的节点集合
    # src_nodes_num = get_src_nodes.shape[0]  # 获取网络中节点个数
    # tar_nodes_num = get_tar_nodes.shape[0]
    print(get_src_nodes.shape)
    print(get_tar_nodes.shape)
    if 1 == 1:
        # source和target网络数据类型由数组array转为list
        tar_edgelist = get_tar.tolist()
        src_edgelist = get_src.tolist()
        start_time = time.time()
        # 网络增强
        # tar_edgelist, src_edgelist = netWork_Enhance(tar_edgelist, src_edgelist, gt[:math.floor(0.8 * len(gt))])
        # 三元闭包
        # tar_edgelist = tri_closure_neg(get_tar_nodes, tar_edgelist)
        # src_edgelist = tri_closure_neg(get_src_nodes, src_edgelist)
        # 去掉叶子节点，负采样的部分也需要修改
        # tar_edgelist,get_tar_nodes = del_leaf_nodes(get_tar_nodes, tar_edgelist)
        # src_edgelist,get_src_nodes = del_leaf_nodes(get_src_nodes, src_edgelist)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"网络结构优化运行时间: {elapsed_time:0.4f} 秒")

        get_src = np.array(src_edgelist)
        get_tar = np.array(tar_edgelist)

    return get_src, get_tar, gt, src_neg_edges, tar_neg_edges, get_src_nodes, get_tar_nodes,gt_tar_add5000,tar_add5000
