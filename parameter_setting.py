# parameter_setting.py

alpha = 0.4  # 重启概率
emb_size = 256  # 嵌入维度
length_walk = 50  # 游走长度
num_walks = 50  # 游走次数
window_size = 10  # Word2Vec的窗口大小
num_iters = 2  # Word2Vec的epochs
train_ratio = 0.8  # 训练集比例
val_ratio = 0.1  # 验证集比例

solution = 23  # 选择使用的游走方式

gt_tar_add5000 = None  # 锚节点数组，用于在融合网络的游走中进行跳转.
