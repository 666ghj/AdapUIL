import numpy as np
import math
from sklearn.metrics import f1_score

def get_statistics(alignment_matrix, groundtruth_matrix):
    pred = greedy_match(alignment_matrix)
    diy_pred = diy_greed_match(alignment_matrix)
    greedy_match_acc,all_acc = compute_accuracy(pred, groundtruth_matrix)
    match_acc,diy_all = compute_accuracy(diy_pred,groundtruth_matrix)
    print("Accuracy: %.4f" % greedy_match_acc)
    # print("DIY_accuracy: %.4f" % match_acc)
    # print("ALL_acc: %.4f" % all_acc)
    # f1 = f1_score(pred, groundtruth_matrix, labels=[0, 1], pos_label=1, average='micro')
    MAP, AUC, Hit = compute_MAP_AUC_Hit(alignment_matrix, groundtruth_matrix)

    print("MAP: %.4f" % MAP)
    print("AUC: %.4f" % AUC)
    print("Hit-precision: %.4f" % Hit)
    # print("f1: %.4f" % f1)

    pred_top_5 = top_k(alignment_matrix, 5)
    precision_5 = compute_precision_k(pred_top_5, groundtruth_matrix)
    print("Precision_5: %.4f" % precision_5)

    pred_top_10 = top_k(alignment_matrix, 10)
    precision_10 = compute_precision_k(pred_top_10, groundtruth_matrix)
    print("Precision_10: %.4f" % precision_10)

    pred_top_15 = top_k(alignment_matrix, 15)
    precision_15 = compute_precision_k(pred_top_15, groundtruth_matrix)
    print("Precision_15: %.4f" % precision_15)

    pred_top_20 = top_k(alignment_matrix, 20)
    precision_20 = compute_precision_k(pred_top_20, groundtruth_matrix)
    print("Precision_20: %.4f" % precision_20)

    pred_top_25 = top_k(alignment_matrix, 25)
    precision_25 = compute_precision_k(pred_top_25, groundtruth_matrix)
    print("Precision_25: %.4f" % precision_25)

    pred_top_30 = top_k(alignment_matrix, 30)
    precision_30 = compute_precision_k(pred_top_30, groundtruth_matrix)
    print("Precision_30: %.4f" % precision_30)

def compute_accuracy(greedy_matched, gt):
    # print(gt)
    n_matched = 0
    m_matched = 0
    for i in range(greedy_matched.shape[0]):
        if greedy_matched[i].sum() > 0 and np.array_equal(greedy_matched[i], gt[i]):
            n_matched += 1
        if np.array_equal(greedy_matched[i],gt[i]):
            m_matched += 1
    n_nodes = (gt==1).sum()
    print('n_nodes')
    print(n_nodes)
    return n_matched/n_nodes,m_matched/greedy_matched.shape[0]


def diy_greed_match(S):
    m,n = S.shape
    result = np.zeros([m, n])
    for index, i in enumerate(S):
        num = np.argsort(-i)
        result[int(index), int(num[0])] = 1
    return result

def greedy_match(S):
    """
    :param S: Scores matrix, shape MxN where M is the number of source nodes,
        N is the number of target nodes.
    :return: A dict, map from source to list of targets.
    """
    S = S.T
    m, n = S.shape

    x = S.T.flatten()
    min_size = min([m,n])
    print(min_size)
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]
        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0: ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result

def compute_MAP_AUC_Hit(alignment_matrix, gt):
    S_argsort = alignment_matrix.argsort(axis=1)[:, ::-1]
    m = gt.shape[1] - 1
    MAP = 0
    AUC = 0
    Hit = 0
    for i in range(len(S_argsort)):
        predicted_source_to_target = S_argsort[i]
        # true_source_to_target = gt[i]
        for j in range(gt.shape[1]):
            if gt[i, j] == 1:
                for k in range(len(predicted_source_to_target)):
                    if predicted_source_to_target[k] == j:
                        ra = k + 1
                        MAP += 1/ra
                        AUC += (m+1-ra)/m
                        Hit += (m+2-ra)/(m+1)
                        break
                break
    n_nodes = (gt==1).sum()
    MAP /= n_nodes
    AUC /= n_nodes
    Hit /= n_nodes
    return MAP, AUC, Hit

def top_k(S, k=1):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    top = np.argsort(-S)[:,:k]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx,elm] = 1
    return result

def compute_precision_k(top_k_matrix, gt):
    n_matched = 0
    gt_candidates = np.argmax(gt, axis = 1)
    for i in range(gt.shape[0]):
        if gt[i][gt_candidates[i]] == 1 and top_k_matrix[i][gt_candidates[i]] == 1:
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes
