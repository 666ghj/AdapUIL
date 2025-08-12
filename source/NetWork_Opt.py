"""
@File    ：NetWork_Opt.py
@Author  : silvan
@Time    : 2024/4/5 17:40

"""
import itertools

def netWork_Enhance(tar_edgelist, src_edgelist, gt):
    """
    网络增强: 对于两个网络t和s，如果在其中一个网络中两个节点x和y存在边，
            同时根据gt这两个节点在另一个网络中也存在x'和y'，那么x'和y'应当也有边
    入参：
        tar_edgelist：目标网络
        src_edgelist：源网络
        gt：ground truth 已知匹配节点对
    返回：
        tar_edgelist： 增强后的目标网络
        src_edgelist： 增强后的源网络
    """
    gt_dict_s_to_t = {}
    gt_dict_t_to_s = {}
    src_gt = set()
    tar_gt = set()

    for src, tar in gt:
        src_gt.add(src)
        tar_gt.add(tar)
        gt_dict_s_to_t[src] = tar
        gt_dict_t_to_s[tar] = src

    for src, tar in tar_edgelist:
        '''
        对象： tar_edgelist：target节点数据集
              tar_gt：groundtruth的target部分
        i[0] in tar_gt and i[1] in tar_gt：
            如果x值在tar_gt and y值也在tar_gt，表示x和y在target里面是连接的
        [gt_dict_t_to_s[i[0]], gt_dict_t_to_s[i[1]]] not in src_edgelist：
            同时，如果x和y根据groundtruth对应的source匹配节点x'和y'在src_edgelist中不存在，表示在source里没有连接
        '''
        if src in tar_gt and tar in tar_gt and [gt_dict_t_to_s[src], gt_dict_t_to_s[tar]] not in src_edgelist:
            # 在source中添加该x'和y'之间的连接
            src_edgelist.append([gt_dict_t_to_s[src], gt_dict_t_to_s[tar]])
    for src, tar in src_edgelist:  # src_edgelist为源节点数据集
        if src in src_gt and tar in src_gt and [gt_dict_s_to_t[src], gt_dict_s_to_t[tar]] not in tar_edgelist:
            tar_edgelist.append([gt_dict_s_to_t[src], gt_dict_s_to_t[tar]])
    return tar_edgelist, src_edgelist

def tri_closure(nodes, edgelist):
    """
    同一个网络里，两个节点有共同邻居，而且他们之间原本没有边，那么增加一条边。
    这个过程需要控制判断有没有边是原始图，这样避免图无限增加边。
    入参：
        nodes 待处理网络提取出的所有不重复节点
        edgelist 待处理网络
    返回：
        edgelist 添加三元闭包的网络
    """
    # new_edges = set()  # 用集合收集需要添加的边
    # for i in nodes:
    #     # indices = find_pairs_with_data(edgelist, i)
    #     adjacency_map = {}
    #     for x, y in edgelist:
    #         adjacency_map.setdefault(x, set()).add(y)
    #         adjacency_map.setdefault(y, set()).add(x)
    #     if i in adjacency_map:
    #         indices = adjacency_map[i]
    #     else:
    #         indices = set()
    #
    #     if len(indices) > 1:
    #         # 使用itertools.combinations生成两两组合
    #         combinations = itertools.combinations(indices, 2)
    #         for pair in combinations:
    #             if pair not in edgelist:
    #                 # 添加该节点对之间的连接
    #                 new_edges.add(pair)
    # edgelist.extend(new_edges)  # 一次性添加所有新的边
    # 转换为集合以便快速查找
    # 初始化新边集合
    new_edges = set()

    # 构建邻接映射
    adjacency_map = {}
    for x, y in edgelist:
        adjacency_map.setdefault(x, set()).add(y)
        adjacency_map.setdefault(y, set()).add(x)

    # 检查每个节点及其邻接点
    for node in nodes:
        neighbors = adjacency_map.get(node, set())
        if len(neighbors) > 1:
            # 生成邻接点之间的组合
            combinations = itertools.combinations(neighbors, 2)
            # 检查哪些组合不是现有的边，并添加到新边集合中
            new_edges.update((edge for edge in combinations if edge not in edgelist))

            # 将新边作为元组添加到edgelist列表中
    edgelist.extend(list(new_edges))
    return edgelist

def tri_closure_neg(nodes, edgelist):
    """
    同一个网络里，两个节点有共同邻居两个以上，而且他们之间原本没有边，那么增加一条边。
    这个过程需要控制判断有没有边是原始图，这样避免图无限增加边。
    入参：
        nodes 待处理网络提取出的所有不重复节点
        edgelist 待处理网络
    返回：
        edgelist 添加三元闭包的网络
    """
    new_edges = set()
    # 建立字典，提高代码效率    为所有节点预先创建空集合
    adjacency_map = {node: set() for node in nodes}
    for x, y in edgelist:
        adjacency_map[x].add(y)
        adjacency_map[y].add(x)

    for i in nodes:
        if i in adjacency_map:
            indices = adjacency_map[i]
        else:
            indices = set()
        if len(indices) > 1:
            combinations = itertools.combinations(indices, 2)
            for pair in combinations:
                # 如果该节点对不存在于edgelist and 该节点对的共同节点数大于等于1
                if pair not in edgelist and len(adjacency_map[pair[0]] & adjacency_map[pair[1]]) >= 1:
                    new_edges.add(pair)
    edgelist.extend(new_edges)
    return edgelist
    # new_edges = set()
    # adjacency_map = {}
    # common_neighbors_map = {}  # 用于存储节点对的共同邻居数
    #
    # # 构建邻接关系映射和节点对的共同邻居数
    # for x, y in edgelist:
    #     adjacency_map.setdefault(x, set()).add(y)
    #     adjacency_map.setdefault(y, set()).add(x)
    #     common_neighbors = len(adjacency_map[x] & adjacency_map[y])
    #     common_neighbors_map[(x, y)] = common_neighbors
    #     common_neighbors_map[(y, x)] = common_neighbors  # 因为是无向图，所以双向都存储
    #
    # # 检查每个节点的邻接节点对
    # for i in nodes:
    #     if i in adjacency_map:
    #         indices = adjacency_map[i]
    #         if len(indices) > 1:
    #             combinations = itertools.combinations(indices, 2)
    #             for pair in combinations:
    #                 if pair not in edgelist and common_neighbors_map[pair] > 2:
    #                     new_edges.add(pair)
    #
    # edgelist.extend(new_edges)
    # return edgelist

def del_leaf_nodes(nodes, edgelist):
    """
        删除叶子节点。
        入参：
            nodes 待处理网络提取出的所有不重复节点
            edgelist 待处理网络
        返回：
            edgelist 删除叶子节点的网络
        """
    # 注释代码效率不高
    '''
    indices1 = set()
    indices2 = set()
    nodes = nodes.tolist()
    for i in nodes:
        indices = set()
        for j, pair in enumerate(edgelist):
            if i in pair:
                indices.add(j)
        if len(indices) == 1:
            # 删除该节点
            indices1.add(next(iter(indices)))
            indices2.add(i)
    for index in sorted(indices1, reverse=True):
        del edgelist[index]
    for index in sorted(indices2, reverse=True):
        del nodes[index]
    '''

    # start_time = time.time()
    adjacency_list = {node: set() for node in nodes}
    for index, (node1, node2) in enumerate(edgelist):
        adjacency_list[node1].add(index)
        adjacency_list[node2].add(index)

        # 遍历邻接表，收集叶子节点
    leaf_nodes = set()
    for node in nodes:
        if len(adjacency_list[node]) == 1:
            leaf_nodes.add(node)
            # 删除叶子节点对应的边
    edgelist = [edge for edge in edgelist if edge[0] not in leaf_nodes and edge[1] not in leaf_nodes]
    # 删除叶子节点
    nodes = [node for node in nodes if node not in leaf_nodes]
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"代码运行时间: {elapsed_time:.6f} 秒")
    return edgelist ,nodes

