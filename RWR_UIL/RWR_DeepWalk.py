import networkx as nx
import numpy as np
import random
from gensim.models import Word2Vec, KeyedVectors
# from .custom_word2vec import Word2Vec
import torch
from scipy.sparse import diags, csr_matrix
from tqdm import tqdm
import ctypes
from ctypes import c_int, c_float, c_double, POINTER

# 游走方式由parameter_setting.py选取
from parameter_setting import solution, gt_tar_add5000


# 定义C语言的接口，纯用AM
lib = ctypes.CDLL('CPython_GeneratingSequences\Only_AffinityMatrix_efficient.dll')
lib.affinity_matrix_walk.argtypes = [
    POINTER(c_float),  # 亲和矩阵
    c_int,             # 矩阵大小
    POINTER(c_int),    # 节点列表
    c_int,             # 节点数量
    c_int,             # 游走长度
    c_int,             # 游走轮数
    POINTER(POINTER(c_int))  # 输出的游走结果
]
lib.affinity_matrix_walk.restype = None

# 定义新的 C 语言接口，AM同时考虑alpha重启
lib_alpha = ctypes.CDLL('CPython_GeneratingSequences/alpha_AffinityMatrix_efficient.dll')
lib_alpha.affinity_matrix_walk_with_restart.argtypes = [
    POINTER(c_float),  # 亲和矩阵
    c_int,             # 矩阵大小
    POINTER(c_int),    # 节点列表
    c_int,             # 节点数量
    c_int,             # 游走长度
    c_int,             # 游走轮数
    c_float,           # 重启概率 alpha
    POINTER(POINTER(c_int))  # 输出的游走结果
]
lib_alpha.affinity_matrix_walk_with_restart.restype = None



# 设置随机种子
torch.manual_seed(616)
random.seed(616)
np.random.seed(616)


class RWR_DeepWalk:
    def __init__(self, node_num, edges, G, alpha, emb_size, length_walk, num_walks,
                 window_size, num_iters, gt_tar_add5000=gt_tar_add5000, device='cuda'):
        """大类方法初始化函数
           内部包含了多种游走方式，包括普通游走、重启游走、带权游走、融合游走等，需要在该类中自行修改游走方式
           该类外部调用train方法，返回w2v向量输出

        Args:
            node_num (_type_): 节点数量
            edges (_type_): 边数量
            G (_type_): NetworkX 库创建的无向图对象
            alpha (float, optional): 重启概率. Defaults to 0.4.
            emb_size (int, optional): 嵌入维度. Defaults to 256.
            length_walk (int, optional): 游走长度. Defaults to 50.
            num_walks (int, optional): 游走轮数. Defaults to 10.
            window_size (int, optional): w2v窗口大小. Defaults to 10.
            num_iters (int, optional): Word2Vec模型的训练迭代次数. Defaults to 1.
            gt_tar_add5000 (np.ndarray, optional): 锚节点数组，用于在融合网络的游走中进行跳转. Defaults to None.
            device (str, optional): 运行设备（'cpu' 或 'cuda'）. Defaults to 'cuda'.
        """
        self.device = torch.device(device)
        if torch.cuda.is_available() and device == 'cuda':
            print("使用GPU进行计算，设备：", torch.cuda.get_device_name(0))
        else:
            print("使用CPU进行计算")
        self.G = G
        self.node_num = node_num
        self.edges = edges
        self.emb_size = emb_size
        self.length_walk = length_walk
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_iters = num_iters
        self.alpha = alpha
        self.gt_tar_add5000 = gt_tar_add5000
        self.adjacency = csr_matrix(nx.adjacency_matrix(self.G))  # 计算邻接矩阵并转换为压缩稀疏行矩阵（CSR）
        self.transition_matrix = self.get_transition_matrix()  # 计算概率转移矩阵，tensor类型，方便后期亲和矩阵计算
        
        
        
    # ---------------------------------------------------以下为工具函数---------------------------------------------------
    
    # 计算概率转移矩阵的函数
    def get_transition_matrix(self):
        print("计算概率转移矩阵")
        degrees = np.array(self.adjacency.sum(axis=1)).flatten()  # 计算每个节点的度
        degrees[degrees == 0] = 1  # 将度为0的节点度设置为1，避免除以0，后面不影响计算0/1结果还是0
        D_inv = diags(1.0 / degrees)  # 构建度的逆对角矩阵
        P = D_inv.dot(self.adjacency)  # 计算概率转移矩阵 P = D^{-1} * A
        return torch.tensor(P.toarray(), dtype=torch.float32).to(self.device)  # 转换为PyTorch张量并移动到GPU
    
    
    # 计算个性化随机游走的亲和矩阵并归一化---计算策略1：从 二阶 概率转移矩阵开始加
    def personalized_random_walk1(self, alpha, max_iter):
        """接收到概率转移矩阵后，计算亲和矩阵并归一化，使用pytorch张量进行计算，提升速度

        Args:
            alpha (float, optional): 重启概率，也可以用在亲和矩阵的计算中. Defaults to 0.4.
            max_iter (int, optional): 理论应该无穷知道收敛，这里默认取1000. Defaults to 1000.

        Returns:
            _type_: 返回一个归一化的亲和矩阵 M ，用于后续游走，tensor类型
        """
        P = self.transition_matrix
        M = torch.zeros_like(P).to(self.device)
        power_P = torch.eye(P.shape[0], device=self.device)  # 创建一个单位矩阵，P^0 = I
        power_P = power_P.matmul(P)
        power_P = power_P.matmul(P)
        
        if alpha == 0:
            for k in tqdm(range(max_iter), desc="亲和矩阵计算中 (alpha=0)"):
                M += power_P  # 当 alpha = 0 时，M 为所有 P^k 的累加，K从2开始
                power_P = power_P.matmul(P)  # 计算 P^k
        else:
            for k in tqdm(range(max_iter), desc="亲和矩阵计算中（alpha!=0）"):
                M += alpha * (1 - alpha) ** (k + 1) * power_P  # 计算亲和矩阵 M
                power_P = power_P.matmul(P)  # 计算 P^k
        # 对亲和矩阵进行归一化
        row_sums = M.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # 避免除以0
        M = M / row_sums
        return M
    
    # 计算个性化随机游走的亲和矩阵并归一化---计算策略2：从 一阶 概率转移矩阵开始加
    def personalized_random_walk2(self, alpha, max_iter):
        """接收到概率转移矩阵后，计算亲和矩阵并归一化，使用pytorch张量进行计算，提升速度

        Args:
            alpha (float, optional): 重启概率，也可以用在亲和矩阵的计算中. Defaults to 0.4.
            max_iter (int, optional): 理论应该无穷知道收敛，这里默认取1000. Defaults to 1000.

        Returns:
            _type_: 返回一个归一化的亲和矩阵 M ，用于后续游走，tensor类型
        """
        P = self.transition_matrix
        M = torch.zeros_like(P).to(self.device)
        power_P = torch.eye(P.shape[0], device=self.device)  # 创建一个单位矩阵，P^0 = I
        power_P = power_P.matmul(P) # 从一阶开始
        
        if alpha == 0:
            for k in tqdm(range(max_iter), desc="亲和矩阵计算中 (alpha=0)"):
                M += power_P  # 当 alpha = 0 时，M 为所有 P^k 的累加，K从2开始
                power_P = power_P.matmul(P)  # 计算 P^k
        else:
            for k in tqdm(range(max_iter), desc="亲和矩阵计算中（alpha!=0）"):
                M += alpha * (1 - alpha) ** k * power_P  # 计算亲和矩阵 M
                power_P = power_P.matmul(P)  # 计算 P^k
        # 对亲和矩阵进行归一化
        row_sums = M.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # 避免除以0
        M = M / row_sums
        return M



    # -------------------------------------------------以下为多种游走方式-------------------------------------------------
    # 注：每种游走方式都返回完整的游走序列集，直接用于后续训练Word2Vec模型，序列中的节点为str类型
    
    # 单纯使用亲和矩阵进行随机游走（效率优化，C混编）
    def RWR_random_walk_AffinityMatrix_efficient(self, G, length_walk, num_walks, alpha, max_iter=1000):
        print("单纯使用亲和矩阵进行随机游走（效率优化，C混编）,alpha=", alpha)
        
        # 计算亲和矩阵
        print("max_iter: ", max_iter)
        M = self.personalized_random_walk1(alpha, max_iter)
        M = M.cpu().numpy() # tensor算完，转成numpy，cpu上方便用
        
        # 准备调用C代码的输入数据
        M_flat = M.flatten().astype(np.float32)  # 将矩阵展平成一维数组
        M_ptr = M_flat.ctypes.data_as(POINTER(c_float))  # 获取指向数组的指针
        nodes = np.array(list(G.nodes()), dtype=np.int32)  # 获取节点列表
        nodes_ptr = nodes.ctypes.data_as(POINTER(c_int))  # 获取指向节点列表的指针
        num_nodes = len(nodes)  # 节点数量
        
        # 准备输出的游走结果
        walks_result = (POINTER(c_int) * (num_walks * num_nodes))()  # 创建指针数组来存储每次游走的结果

        # 调用C函数进行游走
        lib.affinity_matrix_walk(M_ptr, M.shape[0], nodes_ptr, num_nodes, length_walk, num_walks, walks_result)
        
        # 将C函数的输出转换为Python列表
        walks = []
        for i in range(num_walks * num_nodes):
            walk = []
            for j in range(length_walk):
                walk.append(str(walks_result[i][j]))
            walks.append(walk)

        return walks
    
    
    # 单纯使用亲和矩阵进行随机游走（未效率优化）（亲和矩阵可设置重启alpha）
    def RWR_random_walk_AffinityMatrix(self, G, length_walk, num_walks, alpha, max_iter=1000):
        print("单纯使用亲和矩阵进行随机游走（未效率优化）,alpha=", alpha)
        
        # 计算亲和矩阵
        print("max_iter: ", max_iter)
        M = self.personalized_random_walk1(alpha, max_iter)
        M = M.cpu().numpy() # tensor算完，转成numpy，cpu上方便用
        
        # 开始游走
        walks = []
        nodes = list(G.nodes())
        for i in tqdm(range(num_walks), desc="num_walks"):
            for node in nodes:
                # 检查当前节点是否为孤立节点
                if len(G[node]) == 0:
                    # 生成一条全是自己的序列
                    walks.append([str(node)] * length_walk)
                else:
                    walk = [str(node)]
                    current_node = node
                    for j in range(length_walk-1):
                        weights = M[current_node]
                        next_node = random.choices(range(M.shape[0]), weights=weights, k=1)[0]
                        walk.append(str(next_node))
                        current_node = next_node
                    walks.append(walk)         

        return walks


    # 重启（可选）随机游走
    def RWR_random_walk(self, G, length_walk, num_walks, alpha):
        print("重启（可选）随机游走,alpha=", alpha)
        
        # 开始游走
        walks = []
        nodes = list(G.nodes())
        for i in tqdm(range(num_walks), desc="num_walks"):
            for node in nodes:
                # 检查当前节点是否为孤立节点
                if len(G[node]) == 0:
                    # 生成一条全是自己的序列
                    walks.append([str(node)] * length_walk)
                else:
                    walk = [str(node)]
                    current_node = node
                    for j in range(length_walk-1):
                        if random.random() > alpha:
                            next_node = random.choice(list(G.neighbors(current_node)))
                        else:
                            next_node = node
                        walk.append(str(next_node))
                        current_node = next_node
                    walks.append(walk)         

        return walks
        

    # 两网络融合后重启（可选）随机游走
    def RWR_random_walk_union(self, G, length_walk, num_walks, alpha):
        print("两网络融合后重启（可选）随机游走, alpha=", alpha)
        
        # 开始游走
        walks = []
        nodes = list(G.nodes()) # 所有节点
        for _ in tqdm(range(num_walks), desc="num_walks"):
            for node in nodes:
                # 检查当前节点是否为孤立节点
                if len(G[node]) == 0:
                    # 生成一条全是自己的序列
                    walks.append([str(node)] * length_walk)
                else:
                    walk = [str(node)] # 拼接首节点
                    v = node # 将首节点赋值给当前节点v
                    for _ in range(length_walk - 1):
                        nbs = list(self.G.neighbors(v))  # 得到所有邻居节点
                        if len(nbs) == 0:
                            break
                        if np.any(self.gt_tar_add5000 == v):  # 如果当前节点v在gt中
                            Pt = 0.5
                            if random.random() < Pt:  # 以Pt的概率跳转到v的锚节点
                                # 获取 v在gt中的索引，由于 v 在 gt_tar_add5000 中只会出现一次，可以直接使用 np.where
                                index = np.where(self.gt_tar_add5000 == v)[0][0]
                                v = [x for x in self.gt_tar_add5000[index] if x != v][0]
                            else:
                                v = random.choice(nbs)  # 以(1-Pt)/(N-1)的概率跳转到gt中的其他邻居节点
                        else:
                            # 如果v不在gt中，以1/N的概率跳转到任何邻居节点
                            v = random.choice(nbs)
                        walk.append(str(v))
                    walks.append(walk)
                    
        return walks
    

    # 使用personalized_random_walk2的亲和矩阵进行随机游走（效率优化，C混编）
    def RWR_random_walk_AffinityMatrix_efficient2(self, G, length_walk, num_walks, alpha, max_iter=1000):
        print("使用personalized_random_walk2的亲和矩阵进行随机游走（效率优化，C混编）,alpha=", alpha)
        
        # 计算亲和矩阵
        print("max_iter: ", max_iter)
        M = self.personalized_random_walk2(alpha, max_iter)
        M = M.cpu().numpy() # tensor算完，转成numpy，cpu上方便用
        
        # 准备调用C代码的输入数据
        M_flat = M.flatten().astype(np.float32)  # 将矩阵展平成一维数组
        M_ptr = M_flat.ctypes.data_as(POINTER(c_float))  # 获取指向数组的指针
        nodes = np.array(list(G.nodes()), dtype=np.int32)  # 获取节点列表
        nodes_ptr = nodes.ctypes.data_as(POINTER(c_int))  # 获取指向节点列表的指针
        num_nodes = len(nodes)  # 节点数量
        
        # 准备输出的游走结果
        walks_result = (POINTER(c_int) * (num_walks * num_nodes))()  # 创建指针数组来存储每次游走的结果

        # 调用C函数进行游走
        lib.affinity_matrix_walk(M_ptr, M.shape[0], nodes_ptr, num_nodes, length_walk, num_walks, walks_result)
        
        # 将C函数的输出转换为Python列表
        walks = []
        for i in range(num_walks * num_nodes):
            walk = []
            for j in range(length_walk):
                walk.append(str(walks_result[i][j]))
            walks.append(walk)

        return walks
    
    
    # 使用personalized_random_walk1的亲和矩阵+同时重启进行随机游走（效率优化，C混编）（二阶开始）
    def RWR_random_walk_AffinityMatrix_with_restart(self, G, length_walk, num_walks, alpha, max_iter=1000):
        print("使用亲和矩阵+同时重启的随机游走（效率优化，C 混编）（二阶开始），alpha =", alpha)
        
        # 计算亲和矩阵
        print("max_iter: ", max_iter)
        M = self.personalized_random_walk1(0, max_iter) # 设置为0，得到AM后在C代码中设置alpha！
        M = M.cpu().numpy()  # 将 tensor 转换为 numpy 数组，方便后续操作
        
        # 准备调用 C 函数的数据
        M_flat = M.flatten().astype(np.float32)  # 将矩阵展平成一维数组
        M_ptr = M_flat.ctypes.data_as(POINTER(c_float))  # 获取指向数组的指针
        nodes = np.array(list(G.nodes()), dtype=np.int32)  # 获取节点列表
        nodes_ptr = nodes.ctypes.data_as(POINTER(c_int))  # 获取指向节点列表的指针
        num_nodes = len(nodes)  # 节点数量
        
        # 准备输出的游走结果
        walks_result = (POINTER(c_int) * (num_walks * num_nodes))()  # 创建指针数组来存储每次游走的结果

        # 调用新的 C 函数进行游走
        lib_alpha.affinity_matrix_walk_with_restart(
            M_ptr, M.shape[0], nodes_ptr, num_nodes,
            length_walk, num_walks, c_float(alpha), walks_result
        )
        
        # 将 C 函数的输出转换为 Python 列表
        walks = []
        for i in range(num_walks * num_nodes):
            walk = []
            for j in range(length_walk):
                walk.append(str(walks_result[i][j]))
            walks.append(walk)

        return walks
    
    # 使用personalized_random_walk2的亲和矩阵+同时重启进行随机游走（效率优化，C混编）（一阶开始）
    def RWR_random_walk_AffinityMatrix_with_restart2(self, G, length_walk, num_walks, alpha, max_iter=1000):
        print("使用亲和矩阵+同时重启的随机游走（效率优化，C 混编）（一阶开始），alpha =", alpha)
        
        # 计算亲和矩阵
        print("max_iter: ", max_iter)
        M = self.personalized_random_walk2(0, max_iter) # 设置为0，得到AM后在C代码中设置alpha！
        M = M.cpu().numpy()  # 将 tensor 转换为 numpy 数组，方便后续操作
        
        # 准备调用 C 函数的数据
        M_flat = M.flatten().astype(np.float32)  # 将矩阵展平成一维数组
        M_ptr = M_flat.ctypes.data_as(POINTER(c_float))  # 获取指向数组的指针
        nodes = np.array(list(G.nodes()), dtype=np.int32)  # 获取节点列表
        nodes_ptr = nodes.ctypes.data_as(POINTER(c_int))  # 获取指向节点列表的指针
        num_nodes = len(nodes)  # 节点数量
        
        # 准备输出的游走结果
        walks_result = (POINTER(c_int) * (num_walks * num_nodes))()  # 创建指针数组来存储每次游走的结果

        # 调用新的 C 函数进行游走
        lib_alpha.affinity_matrix_walk_with_restart(
            M_ptr, M.shape[0], nodes_ptr, num_nodes,
            length_walk, num_walks, c_float(alpha), walks_result
        )
        
        # 将 C 函数的输出转换为 Python 列表
        walks = []
        for i in range(num_walks * num_nodes):
            walk = []
            for j in range(length_walk):
                walk.append(str(walks_result[i][j]))
            walks.append(walk)

        return walks

    # --------------------------------------以下为外部使用方法，在这里修改使用的游走方式-------------------------------------
    
    def train(self, workers=4, is_loadmodel=False, is_loaddata=False, node_type="source", solution=solution):
        """
        Args:
            solution：选择使用的游走组合方式
                      - 0.1：RWR_random_walk
                      - 0.2：RWR_random_walk_AffinityMatrix_efficient
                      - 0.3：RWR_random_walk_union
                      - 1：RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(不重启)；结果1：1拼接
                      - 2：RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(重启概率与前者相同)；结果1：1拼接
                      - 3：RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(重启概率与前者相同)；结果1：1加和
                      - 4：RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(重启概率与前者相同)；结果0.8：0.2加和
                      - 5: RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(重启概率与前者相同)；结果0.5：0.5加和
                      - 6: RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(重启概率与前者相同)；结果0.2：0.8加和
                      - 7: RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(不重启，但是使用第二个亲和矩阵计算方式游走)；结果1：1拼接
                      - 8：RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(重启概率与前者相同，但是使用第二个亲和矩阵计算方式游走)；结果1：1拼接
                      - 9：RWR_random_walk + RWR_random_walk(重启概率与前者相同)；结果1：1拼接
                      - 10：RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(alpha=0.7，但是使用第二个亲和矩阵计算方式游走)；结果1：1拼接
                      - 11：RWR_random_walk + RWR_random_walk_AffinityMatrix_efficient(alpha=0.15，但是使用第二个亲和矩阵计算方式游走)；结果1：1拼接
        Returns:
            w2v: 给下游任务使用的嵌入向量，由Word2Vec训练得到，注意一定要是Word2Vec对象（不可以是字典）
        """
        if solution == 0.1:
            print("node_type: ",node_type)
            # 打印告知当前使用的是哪种方法
            print('当前使用的游走方式是：RWR_random_walk')
            walks = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            print('Number of sentenses to train: ', len(walks))
            print('Start training...')
            w2v = Word2Vec(
                sentences=walks,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print('Training Done.###################################################')
            return w2v
        
        elif solution == 0.2:
            print("node_type: ",node_type)
            # 打印告知当前使用的是哪种方法
            print('当前使用的游走方式是：RWR_random_walk_AffinityMatrix_efficient')
            walks = self.RWR_random_walk_AffinityMatrix_efficient(self.G, self.length_walk, self.num_walks, self.alpha)
            print('Number of sentenses to train: ', len(walks))
            print('Start training...')
            w2v = Word2Vec(
                sentences=walks,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print('Training Done.###################################################')
            return w2v
            
        elif solution == 0.3:
            print("node_type: ",node_type)
            # 打印告知当前使用的是哪种方法
            print('当前使用的游走方式是：RWR_random_walk_union')
            walks = self.RWR_random_walk_union(self.G, self.length_walk, self.num_walks, self.alpha)
            print('Number of sentenses to train: ', len(walks))
            print('Start training...')
            w2v = Word2Vec(
                sentences=walks,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print('Training Done.###################################################')
            return w2v
            
        elif solution == 1:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient(self.G, self.length_walk, self.num_walks, alpha=0)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 2:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient(self.G, self.length_walk, self.num_walks, alpha=self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 3:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient(self.G, self.length_walk, self.num_walks, alpha=self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # w2v1.wv.save_word2vec_format("w2v1.txt")
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # w2v2.wv.save_word2vec_format("w2v2.txt")
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1加和###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = w2v1.wv[word] + w2v2.wv[word]
            # w2v.wv.save_word2vec_format("w2v.txt")
            return w2v
        
        elif solution == 4:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient(self.G, self.length_walk, self.num_walks, alpha=self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始0.8：0.2加和###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = 0.8*w2v1.wv[word] + 0.2*w2v2.wv[word]
            return w2v
        
        elif solution == 5:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient(self.G, self.length_walk, self.num_walks, alpha=self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始0.5：0.5加和###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = 0.5*w2v1.wv[word] + 0.5*w2v2.wv[word]
            return w2v
        
        elif solution == 6:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient(self.G, self.length_walk, self.num_walks, alpha=self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始0.2：0.8加和###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = 0.2*w2v1.wv[word] + 0.8*w2v2.wv[word]
            return w2v
        
        elif solution == 7:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, alpha=0)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 8:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 9:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 10:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, alpha=0.7)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 11:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, alpha=0.15)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 12:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, alpha=0.4)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 13:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, alpha=0.4)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=128,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=32,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 14:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, alpha=0.4)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=32,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=128,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 15:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, alpha=0.4)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=64,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=128,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 16:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, alpha=0.4)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=128,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=64,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 17:
            walks1 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_union(self.G, self.length_walk, self.num_walks, self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 18:
            walks1 = self.RWR_random_walk_union(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 19:
            walks1 = self.RWR_random_walk_union(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks3 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=self.emb_size,  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks3开始Word2Vec")
            w2v3 = Word2Vec(
                sentences=walks3,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = w2v1.wv[word] + np.concatenate((w2v2.wv[word], w2v3.wv[word]))
            return w2v
        
        elif solution == 20:
            walks1 = self.RWR_random_walk_union(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk(self.G, self.length_walk, self.num_walks, self.alpha)
            walks3 = self.RWR_random_walk_AffinityMatrix_efficient2(self.G, self.length_walk, self.num_walks, self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/4),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks3开始Word2Vec")
            w2v3 = Word2Vec(
                sentences=walks3,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/4),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word], w2v3.wv[word]))
            return w2v
        
        elif solution == 21:
            walks1 = self.RWR_random_walk_union(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_efficient(self.G, self.length_walk, self.num_walks, self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 22:
            walks1 = self.RWR_random_walk_union(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_with_restart(self.G, self.length_walk, self.num_walks, self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        elif solution == 23:
            walks1 = self.RWR_random_walk_union(self.G, self.length_walk, self.num_walks, self.alpha)
            walks2 = self.RWR_random_walk_AffinityMatrix_with_restart2(self.G, self.length_walk, self.num_walks, self.alpha)
            print("walks1开始Word2Vec")
            w2v1 = Word2Vec(
                sentences=walks1,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            print("walks2开始Word2Vec")
            w2v2 = Word2Vec(
                sentences=walks2,     # 训练数据，节点游走序列集
                vector_size=int(self.emb_size/2),  # 嵌入向量维度大小
                window=self.window_size,   # 上下文窗口大小
                epochs=self.num_iters,    # 训练的迭代次数
                sg=1,                     # 训练算法：1为Skip-gram，0为CBOW
                hs=1,                     # 是否使用层次Softmax：1为是，0为否（使用负采样）
                min_count=0,              # 忽略所有频率低于此值的词，这里为0表示不忽略任何词
                workers=workers           # 用于训练的线程数
            )
            # 提取并拼接嵌入结果
            print('Training Done.开始1：1拼接###################################################')
            # 创建一个新的 Word2Vec 模型
            w2v = Word2Vec(vector_size=self.emb_size, window=self.window_size, sg=1, hs=1, min_count=0, workers=workers)
            all_words = w2v1.wv.index_to_key
            # 构造词汇表和词向量
            w2v.build_vocab([all_words], update=False)
            for word in all_words:
                w2v.wv[word] = np.concatenate((w2v1.wv[word], w2v2.wv[word]))
            return w2v
        
        else:
            print("请在solution参数中选择正确的游走方式")
            return None
