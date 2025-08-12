# sequence_counter.py

import ast
import networkx as nx

def count_sequences(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # 将字符串转换为列表
    sequences = ast.literal_eval(content)

    return sequences

def count_graph_properties(sequences):
    # 创建有向图
    graph = nx.DiGraph()

    # 添加序列中的节点和边
    for seq_index, sequence in enumerate(sequences, start=1):
        for i in range(len(sequence) - 1):
            source_node = sequence[i]
            target_node = sequence[i + 1]
            graph.add_edge(source_node, target_node)

    # 计算序列总数、边的总数和节点总数
    num_sequences = len(sequences)
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    return num_sequences, num_edges, num_nodes

