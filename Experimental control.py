import os
import subprocess

# 创建保存结果的目录
result_dir = "1120实验"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 不变参数
length_walk = 50
num_walks = 50
window_size = 10
num_iters = 2   
train_ratio = 0.8
val_ratio = 0.1

# 方法对应的参数设置
methods = [
    {"method": 4, "alpha": 0.4, "solution": 22, "emb_size": 256},
    {"method": 5, "alpha": 0.4, "solution": 23, "emb_size": 256},
]

# 修改parameter_setting.py文件
def update_parameters(alpha, solution, emb_size):
    with open("parameter_setting.py", "w") as f:
        f.write(f"""# parameter_setting.py

alpha = {alpha}  # 重启概率
emb_size = {emb_size}  # 嵌入维度
length_walk = {length_walk}  # 游走长度
num_walks = {num_walks}  # 游走次数
window_size = {window_size}  # Word2Vec的窗口大小
num_iters = {num_iters}  # Word2Vec的epochs
train_ratio = {train_ratio}  # 训练集比例
val_ratio = {val_ratio}  # 验证集比例

solution = {solution}  # 选择使用的游走方式

gt_tar_add5000 = None  # 锚节点数组，用于在融合网络的游走中进行跳转.
""")

# 执行实验并保存结果
def run_experiment(method_id):
    for i in range(1, 6):
        print(f"正在运行方法 {method_id}, 第 {i} 次实验...")
        # 运行实验
        result = subprocess.run(["python", "RWR_UIL.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # 打印输出到控制台
        print(result.stdout)
        
        # 保存结果
        method_dir = os.path.join(result_dir, f"方法{method_id}")
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)
        
        result_file = os.path.join(method_dir, f"{i}.txt")
        with open(result_file, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

# 批量运行所有方法
for method in methods:
    method_id = method["method"]
    alpha = method["alpha"]
    solution = method["solution"]
    emb_size = method["emb_size"]
    
    print(f"开始运行方法 {method_id} 的实验...")
    update_parameters(alpha, solution, emb_size)
    run_experiment(method_id)
    print(f"方法 {method_id} 的实验运行完毕。")
