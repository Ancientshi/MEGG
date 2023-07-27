import numpy as np
from concurrent.futures import ThreadPoolExecutor

def calculate_precision_recall_f1(test_truth_list, test_prediction_list, topk):
    precisions = []
    recalls = []
    f1_scores = []
    for k in topk:
        precision_list = []
        recall_list = []
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            precision_dem = k
            recall_dem = len(test_truth_index)
            top_sorted_index = set(test_prediction_list[ind][0:k])
            hit_num = len(top_sorted_index.intersection(test_truth_index))
            precision_list.append(hit_num * 1.0 / (precision_dem + 1e-20))
            recall_list.append(hit_num * 1.0 / (recall_dem + 1e-20))
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1_score = 2 * precision * recall / (precision + recall + 1e-20)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    return precisions, recalls, f1_scores


def calculate_mrr(test_truth_list, test_prediction_list, topk):
    mrrs = []
    for k in topk:
        mrr_list = []
        for ind, test_truth in enumerate(test_truth_list):
            mrr = 1.0
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            top_sorted_index = set(test_prediction_list[ind][0:k])
            ctr = 1e20
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    ctr = index + 1
                    break
            mrr /= ctr
            mrr_list.append(mrr)
        mrrs.append(np.mean(mrr_list))
    return mrrs


def calculate_ndcg(test_truth_list, test_prediction_list, topk):
    ndcgs = []
    for k in topk:
        ndcg_list = []
        for ind, test_truth in enumerate(test_truth_list):
            dcg = 0
            idcg = 0
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            top_sorted_index = set(test_prediction_list[ind][0:k])
            idcg_dem = 0
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    dcg += 1.0 / np.log2(index + 2)
                    idcg += 1.0 / np.log2(idcg_dem + 2)
                    idcg_dem += 1
            ndcg = dcg * 1.0 / (idcg + 1e-20)
            ndcg_list.append(ndcg)
        ndcgs.append(np.mean(ndcg_list))
    return ndcgs


# def calculate_all(test_truth_list, test_prediction_list, topk):
#     precisions, recalls, f1_scores = calculate_precision_recall_f1(test_truth_list, test_prediction_list, topk)
#     mrrs = calculate_mrr(test_truth_list, test_prediction_list, topk)
#     ndcgs = calculate_ndcg(test_truth_list, test_prediction_list, topk)
#     return precisions, recalls, f1_scores, mrrs, ndcgs

def calculate_all(test_truth_list, test_prediction_list, topk):
    with ThreadPoolExecutor() as executor:
        precision_recall_f1_future = executor.submit(calculate_precision_recall_f1, test_truth_list, test_prediction_list, topk)
        mrr_future = executor.submit(calculate_mrr, test_truth_list, test_prediction_list, topk)
        ndcg_future = executor.submit(calculate_ndcg, test_truth_list, test_prediction_list, topk)
        
        precisions, recalls, f1_scores = precision_recall_f1_future.result()
        mrrs = mrr_future.result()
        ndcgs = ndcg_future.result()
        
    return precisions, recalls, f1_scores, mrrs, ndcgs


# import numpy as np
# import concurrent.futures

# # 修改每个单独的计算函数，让他们接收单一的输入
# def calculate_precision_recall_f1_single(test_truth, test_prediction, k):
#     test_truth_index = set(test_truth)
#     if len(test_truth_index) == 0:
#         return None
#     precision_dem = k
#     recall_dem = len(test_truth_index)
#     top_sorted_index = set(test_prediction[0:k])
#     hit_num = len(top_sorted_index.intersection(test_truth_index))
#     precision = hit_num * 1.0 / (precision_dem + 1e-20)
#     recall = hit_num * 1.0 / (recall_dem + 1e-20)
#     f1_score = 2 * precision * recall / (precision + recall + 1e-20)
#     return precision, recall, f1_score

# def calculate_mrr_single(test_truth, test_prediction, k):
#     mrr = 1.0
#     test_truth_index = set(test_truth)
#     if len(test_truth_index) == 0:
#         return None
#     top_sorted_index = set(test_prediction[0:k])
#     ctr = 1e20
#     for index, itemid in enumerate(top_sorted_index):
#         if itemid in test_truth_index:
#             ctr = index + 1
#             break
#     mrr /= ctr
#     return mrr

# def calculate_ndcg_single(test_truth, test_prediction, k):
#     dcg = 0
#     idcg = 0
#     test_truth_index = set(test_truth)
#     if len(test_truth_index) == 0:
#         return None
#     top_sorted_index = set(test_prediction[0:k])
#     idcg_dem = 0
#     for index, itemid in enumerate(top_sorted_index):
#         if itemid in test_truth_index:
#             dcg += 1.0 / np.log2(index + 2)
#             idcg += 1.0 / np.log2(idcg_dem + 2)
#             idcg_dem += 1
#     ndcg = dcg * 1.0 / (idcg + 1e-20)
#     return ndcg

# # 用线程池并行计算每个任务
# def calculate_all(test_truth_list, test_prediction_list, topk):
#     all_precisions, all_recalls, all_f1_scores, all_mrrs, all_ndcgs = [], [], [], [], []

#     for k in topk:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future_to_result = {executor.submit(calculate_precision_recall_f1_single, test_truth, test_prediction, k): (test_truth, test_prediction) for test_truth, test_prediction in zip(test_truth_list, test_prediction_list)}
#             precisions, recalls, f1_scores = [], [], []
#             for future in concurrent.futures.as_completed(future_to_result):
#                 precision, recall, f1_score = future.result()
#                 if precision is not None:
#                     precisions.append(precision)
#                     recalls.append(recall)
#                     f1_scores.append(f1_score)

#             future_to_result = {executor.submit(calculate_mrr_single, test_truth, test_prediction, k): (test_truth, test_prediction) for test_truth, test_prediction in zip(test_truth_list, test_prediction_list)}
#             mrrs = []
#             for future in concurrent.futures.as_completed(future_to_result):
#                 mrr = future.result()
#                 if mrr is not None:
#                     mrrs.append(mrr)

#             future_to_result = {executor.submit(calculate_ndcg_single, test_truth, test_prediction, k): (test_truth, test_prediction) for test_truth, test_prediction in zip(test_truth_list, test_prediction_list)}
#             ndcgs = []
#             for future in concurrent.futures.as_completed(future_to_result):
#                 ndcg = future.result()
#                 if ndcg is not None:
#                     ndcgs.append(ndcg)

#         all_precisions.append(np.mean(precisions))
#         all_recalls.append(np.mean(recalls))
#         all_f1_scores.append(np.mean(f1_scores))
#         all_mrrs.append(np.mean(mrrs))
#         all_ndcgs.append(np.mean(ndcgs))
        
#     return all_precisions, all_recalls, all_f1_scores, all_mrrs, all_ndcgs
