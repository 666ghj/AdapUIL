"""
WGW.py
重启随机游走RWR:Random Walk with Restart
去边游走ERW：Edge-Removing Walk，在游走过程中，ERW会拆除已经游走的边，从而完全避免序列的重复；
           同时为了保证节点特征的完整性，ERW会对图进行多个轮次的游走，以便获得某个节点作为起始节点的多种不同可能的序列。
WGW：Deepwalk的random walk替换为ERW
"""

import torch.nn.functional as F
import numpy as np
import torch
import time
import math
import os

from RWR_UIL.RWR_get_stg import RWR_get_stg
from RWR_UIL.RWR_source import RWR_source
from RWR_UIL.RWR_target import RWR_target
from RWR_UIL.RWR_union import RWR_union
from source.sequence_counter import count_graph_properties, count_sequences
from source.evaluate import get_statistics
from source.mapping_model import PaleMappingMlp

# 参数统一由parameter_setting.py进行设置
from parameter_setting import alpha, emb_size, length_walk, num_walks, window_size, num_iters, train_ratio, val_ratio

star_time = time.time()
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
np.random.seed(616)
torch.manual_seed(616)  # 为CPU设置随机种子
torch.cuda.manual_seed(616)  # 为当前GPU设置随机种子

########################### 随机游走参数设置#######################################
# alpha = 0.4  # 重启概率
# emb_size = 256  # 嵌入维度
# length_walk = 50  # 游走长度
# num_walks = 50  # 游走次数
# window_size = 10  # Word2Vec的窗口大小
# num_iters = 2  # Word2Vec的epochs
######################################获取向量表示##############################################
src_edges, tar_edges, gt, src_neg_edges, tar_neg_edges, src_nodes, tar_nodes, gt_tar_add5000, tar_add5000 = RWR_get_stg()  # 获取图数据
# 将两网络合并进行随机游走，获取更高阶网络结构信息
# src_union_vector, tar_union_vector = RWR_union(src_edges, tar_add5000, gt_tar_add5000, alpha, emb_size, length_walk,
#                                                num_walks, window_size, num_iters, union)
# 源图和目标图的向量表示，重启随机游走参数为0，即正常的随机游走，重在获取远端信息
source_vector1 = RWR_source(src_edges, gt, src_neg_edges, alpha, emb_size, length_walk, num_walks, window_size, num_iters)
target_vector1 = RWR_target(tar_edges, gt, tar_neg_edges, alpha, emb_size, length_walk, num_walks, window_size, num_iters)
# 重启随机游走参数为0.2，重在获取近端信息
# source_vector2 = RWR_source(src_edges, gt, src_neg_edges, alpha, emb_size, length_walk, num_walks, window_size,num_iters)
# target_vector2 = RWR_target(tar_edges, gt, tar_neg_edges, alpha, emb_size, length_walk, num_walks, window_size,num_iters)

# source_vector = np.concatenate((source_vector1, src_union_vector), axis=1)
# target_vector = np.concatenate((target_vector1, tar_union_vector), axis=1)

# 直接叠加，效果会不如concate方式，但是不用增加Embedding size
source_vector = source_vector1 #+ src_union_vector
target_vector = target_vector1 #+ tar_union_vector

# MLP的嵌入维度，也是拼接后的向量维度
# embedding_dim = source_vector.shape[1]

####################################数据处理######################################################
source_vector = torch.FloatTensor(source_vector)  # 类型转换, 将list ,numpy转化为tensor
# F.normalize：将某一个维度除以那个维度对应的范数(默认是2范数)，范数是平方求和开根号
# dim:0表示按列操作，则每列都是除以该列下平方和的开方；1表示按行操作，则每行都是除以该行下所有元素平方和的开方
# https://blog.csdn.net/lj2048/article/details/118115681
source_vector = F.normalize(source_vector, dim=1)
source_vector = source_vector.numpy()

target_vector = torch.FloatTensor(target_vector)
target_vector = F.normalize(target_vector, dim=1)
target_vector = target_vector.numpy()


map_act = 'relu'
map_lr = 0.001
map_batch_size = 32
source_vector = torch.FloatTensor(source_vector)
target_vector = torch.FloatTensor(target_vector)
source_vector = source_vector.cuda()
target_vector = target_vector.cuda()
# viz = visdom.Visdom()   # 可视化工具
print("Use Mpl mapping")

########################################映射对齐############################################
mapping_model = PaleMappingMlp(
    embedding_dim=emb_size,  # 如果拼接需要*2
    source_embedding=source_vector,
    target_embedding=target_vector,
    activate_function=map_act,
)

mapping_model = mapping_model.cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mapping_model.parameters()), lr=map_lr)
# np.random.shuffle(gt)
groundtruth = gt.tolist()
groundtruth_source = []
groundtruth_target = []
groundtruth_dict = {}
id2idx_source = {}
id2idx_target = {}
for i in groundtruth:
    groundtruth_source.append(i[0])
    groundtruth_target.append(i[1])
    groundtruth_dict[i[0]] = i[1]
for index, value in enumerate(groundtruth):
    id2idx_source[value[0]] = index
    id2idx_target[value[1]] = index
# 训练集比例
train_size = math.floor(train_ratio * len(groundtruth_source))
val_size = math.floor(val_ratio * len(groundtruth_source))
source_train_nodes = groundtruth_source[:train_size]
source_val_nodes = groundtruth_source[train_size:train_size+val_size]
source_test_nodes = groundtruth_source[train_size:]
np.random.shuffle(source_train_nodes)
np.random.shuffle(source_val_nodes)
np.random.shuffle(source_test_nodes)

n_iters = len(source_train_nodes) // map_batch_size
assert n_iters > 0, "batch_size is too large"
if (len(source_train_nodes) % map_batch_size > 0):
    n_iters += 1
print_every = int(n_iters / 4) + 1
total_steps = 0

for epoch in range(1, 200):
    # np.random.shuffle(source_t    rain_nodes)
    # start = time.time()
    # print('Epochs: ', epoch)
    for iter in range(n_iters):
        source_batch = source_train_nodes[iter * map_batch_size:(iter + 1) * map_batch_size]
        target_batch = [groundtruth_dict[x] for x in source_batch]

        source_batch = torch.LongTensor(source_batch)
        target_batch = torch.LongTensor(target_batch)
        source_batch = source_batch.cuda()
        target_batch = target_batch.cuda()
        optimizer.zero_grad()
        # start_time = time.time()

        loss = mapping_model.loss(source_batch, target_batch)
        loss.backward()
        optimizer.step()

        # if total_steps % print_every == 0 and total_steps > 0:
        #     print("Iter:", '%03d' % iter,
        #           "train_loss=", "{:.5f}".format(loss.item())
        #           # ,"time", "{:.5f}".format(time.time() - start_time)
        #           )
        total_steps += 1

source_after_mapping = mapping_model(source_vector)
S = torch.matmul(source_after_mapping, target_vector.t())
S = S.detach().cpu().numpy()

source_after_mapping = source_after_mapping.detach().cpu().numpy()
target_vector = target_vector.cpu().numpy()

source_rows = source_after_mapping.shape[0]  # 获取source的node数量
target_rows = target_vector.shape[0]  # 获取target的node数量
gt = np.zeros((source_rows, target_rows))
gt2 = np.zeros((source_rows, target_rows))

for i in source_train_nodes:
    gt[i, groundtruth_dict[i]] = 1

for i in source_test_nodes:
    gt2[i, groundtruth_dict[i]] = 1

get_statistics(S, gt)
get_statistics(S, gt2)

end_time = time.time()
execution_time = end_time - star_time

print("Time:", round(execution_time, 2))
'''
source_path = "dataspace/douban/source.txt"
target_path = "dataspace/douban/target.txt"

source_sequences = count_sequences(source_path)
target_sequences = count_sequences(target_path)

num_source_sequences, num_source_edges, num_source_nodes = count_graph_properties(source_sequences)
num_target_sequences, num_target_edges, num_target_nodes = count_graph_properties(target_sequences)

num_sequences = num_source_sequences + num_target_sequences
num_edges = num_source_edges + num_target_edges
num_nodes = num_source_nodes + num_target_nodes

print("序列总数:", num_sequences)
print("边的总数:", num_edges)
print("节点总数:", num_nodes)'''
