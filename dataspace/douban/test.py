def count_unique_nodes(file_path):
    unique_nodes = set()

    try:
        with open(file_path, 'r') as file:
            for line in file:
                node1, node2 = line.split()
                unique_nodes.add(node1)
                unique_nodes.add(node2)

        print(f"不同节点的数量: {len(unique_nodes)}")

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"处理文件时出现错误: {e}")

# 调用函数，假设文件名为 'twitter.edges'
count_unique_nodes('offline.edgelist')
