import sys

import utility.metrics as metrics
from utility.parser import parse_args
from utility.load_data import *
import multiprocessing
import heapq
import torch
from utility.split_dataset import load_test_dict

cores = multiprocessing.cpu_count()
print(cores)
args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USER_NUM, GROUP_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_groups, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
# user_test_dict = load_test_dict(path=args.data_path + args.dataset)
# print(user_test_dict)
# sys.exit()
BATCH_SIZE = args.batch_size


def ranklist_by_heapq(user_pos_test, test_groups, rating, Ks):
    item_score = {}
    for i in test_groups:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_group_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_group_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def get_auc(group_score, user_pos_test):
    group_score = sorted(group_score.items(), key=lambda kv: kv[1])
    group_score.reverse()
    group_sort = [x[0] for x in group_score]
    posterior = [x[1] for x in group_score]

    r = []
    for i in group_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc


def ranklist_by_sorted(user_pos_test, test_groups, rating, Ks):
    group_score = {}
    for i in test_groups:
        group_score[i] = rating[i]

    K_max = max(Ks)
    K_max_group_score = heapq.nlargest(K_max, group_score, key=group_score.get)

    r = []
    for i in K_max_group_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(group_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def get_performance_recall_ndcg(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))

    return {'recall': np.array(recall),
            'ndcg': np.array(ndcg)}


def test_one_user(x):
    # user u's ratings for groups
    rating = x[0]
    # uid
    u = x[1]
    # user u's groups in the training set
    try:
        training_groups = data_generator.train_group[u]
    except Exception:
        training_groups = []
    # user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_groups = set(range(GROUP_NUM))

    test_groups = list(all_groups - set(training_groups))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_groups, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_groups, rating, Ks)
    return get_performance_recall_ndcg(user_pos_test, r, auc, Ks)

def test_one_user_uid(x):
    # user u's ratings for groups
    rating = x[0]
    # uid
    u = x[1]
    # user u's groups in the training set
    try:
        training_groups = data_generator.train_group[u]
    except Exception:
        training_groups = []
    # user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_groups = set(range(GROUP_NUM))

    test_groups = list(all_groups - set(training_groups))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_groups, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_groups, rating, Ks)
    return (u,get_performance_recall_ndcg(user_pos_test, r, auc, Ks))

def test(users_to_test, embedding_h, user_split=False):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}
    recall_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    count_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    ndcg_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    pool = multiprocessing.Pool(cores)

    u_batch_size = args.batch_size

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1 if n_test_users % u_batch_size != 0 else n_test_users // u_batch_size

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        print('test batch %d' % u_batch_id)
        test_users_embedding = embedding_h['user'][user_batch]
        full_groups_embedding = embedding_h['group']
        rate_batch = torch.matmul(test_users_embedding, full_groups_embedding.t()).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
        if user_split:
            print('no split')
            sys.exit(0)
            # uid_batch_result = pool.map(test_one_user_uid, user_batch_rating_uid)
            # count += len(uid_batch_result)
            # for re in uid_batch_result:
            #     # result['precision'] += re['precision'] / n_test_users
            #     user_part = user_test_dict[str(re[0])]
            #     recall = re[1]['recall']
            #     ndcg = re[1]['ndcg']
            #     result['recall'] += recall / n_test_users
            #     result['ndcg'] += ndcg / n_test_users
            #     count_dict[user_part] += 1
            #     recall_dict[user_part] += recall
            #     ndcg_dict[user_part] += ndcg
        else:
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                # result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                # result['hit_ratio'] += re['hit_ratio'] / n_test_users
                # result['auc'] += re['auc'] / n_test_users
    assert count == n_test_users
    pool.close()
    if user_split:
        print(count_dict)
        for key in recall_dict.keys():
            recall_dict[key] = recall_dict[key] / count_dict[key]

        for key in ndcg_dict.keys():
            ndcg_dict[key] = ndcg_dict[key] / count_dict[key]

    return result, recall_dict, ndcg_dict
