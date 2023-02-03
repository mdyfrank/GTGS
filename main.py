import sys

import torch
import torch.optim as optim
from utility.load_data import *
from utility.parser import *
from utility.batch_test import *
from utility.helper import early_stopping, random_batch_users, ensureDir, convert_dict_list, convert_list_str
from model import *
from time import time
import numpy as np
import os


def main(args):
    # Step 1: Prepare graph data and device ================================================================= #
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    users_to_test = list(data_generator.test_set.keys())
    g = data_generator.g
    g = g.to(device)

    pos_g = construct_user_group_bigraph(g)
    if args.gat != 0:
        model = GAT(args, g, args.embed_size, 8, args.embed_size, num_heads=2).to(device)
    else:
        model = LightGCN(args, g, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    t0 = time()
    cur_best_pre_0, stopping_step = 0, 0
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    recall_split_loger, ndcg_split_loger = [], []
    for epoch in range(args.epoch):
        t1 = time()
        neg_g = construct_negative_graph(g, args.neg_samples, device=device)
        embedding_h = model(g)
        bpr_loss, mf_loss, emb_loss = model.create_bpr_loss(pos_g, neg_g, embedding_h, g)
        ssl_loss = 0
        if args.contrastive_learning:
            if args.contrastive_learning!= -1:
                ssl_loss =  model.create_ssl_loss_user(args.ssl_temp)
            if args.contrastive_learning in [-1,2]:
                if 'steam' not in args.dataset:
                    ssl_loss += model.create_ssl_loss_group(args.ssl_temp)
                else:
                    k = 4
                    for idx in range(k):
                        ssl_loss += model.create_ssl_loss_batched_group(args.ssl_temp, k, idx)
            ssl_loss *= args.ssl_reg
            bpr_loss += ssl_loss

        optimizer.zero_grad()
        bpr_loss.backward()
        optimizer.step()
        if (epoch + 1) % (args.verbose * 10) != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f +%.5f]' % (
                    epoch, time() - t1, bpr_loss, mf_loss, emb_loss, ssl_loss)
                print(perf_str)
            continue
        t2 = time()
        if args.fast_test:
            test_users = random_batch_users(users_to_test, args.batch_size)
        else:
            test_users = users_to_test
        ret, recall_dict, ndcg_dict = test(test_users, embedding_h, user_split=args.user_split)
        t3 = time()


        loss_loger.append(bpr_loss)
        rec_loger.append(ret['recall'])
        ndcg_loger.append(ret['ndcg'])
        # align_loger.append(a_score)
        # uniform_u_loger.append(u_uscore)
        # uniform_g_loger.append(u_gscore)

        if args.user_split:
            recall_split_loger.append(convert_dict_list(recall_dict))
            ndcg_split_loger.append(convert_dict_list(ndcg_dict))
        # hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, bpr_loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=args.flag_step)

        # early stop
        if should_stop == True:
            break

    if args.save_flag == 1:
        group_emb = model.h_group_v1.cpu().detach().numpy()
        np.save(args.weights_path + args.dataset, group_emb)
        # user_emb = model.h_user_v1.cpu().detach().numpy()
        # torch.save(model.state_dict(), args.weights_path + args.model_name)
        print('save the weights in path: ', args.weights_path + args.dataset)

    a_score = model.alignment_score()
    u_uscore = model.uniformity_score_user()
    if 'steam' not in args.dataset:
        u_gscore = model.uniformity_score_group()
    else:
        u_gscore = model.uniformity_score_group_batch()
    print(a_score)
    print(u_uscore)
    print(u_gscore)

    recs = np.array(rec_loger)
    # pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    # hit = np.array(hit_loger)
    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s], ssl_metric=[%.5f, %.5f, %.5f]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]),a_score,u_uscore,u_gscore)
    print(final_perf)

    save_path = './output/%s/%s.result' % (args.dataset, args.model_name)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'top_k=%s, lr=%.4f, layer_num=%s, batch_size=%d, norm=%.1f, gat=%d, lightgcn_iu=%d,  hgcn=%d, hgcn_ug_side=%d, hgcn_u_hyperedge=%d, pre_gcn=%d, hgcn_mix=%.4f, user_hpedge_ig=%d, contrastive_learning=%d, ssl_reg=%.7f, ssl_temp=%.2f, reweight_type=%d, beta_group=%.2f, beta_item=%.2f,regs=%s\n\t%s\n'
        % (args.Ks, args.lr, args.layer_num, args.batch_size, args.norm_2, args.gat, args.lightgcn_iu, args.hgcn,
           args.hgcn_ug_side,
           args.hgcn_u_hyperedge,
           args.pre_gcn,args.hgcn_mix,
           args.user_hpedge_ig, args.contrastive_learning, args.ssl_reg, args.ssl_temp, args.reweight_type,
           args.beta_group, args.beta_item,
           args.regs, final_perf))
    f.close()
    if args.user_split == 1:
        final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % \
                     (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                      '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
        final_perf += '\n\trecalls: ' + convert_list_str(recall_split_loger[idx]) + '\n'
        final_perf += '\tndcgs:   ' + convert_list_str(ndcg_split_loger[idx]) + '\n'
        save_path_split = './output/%s/%s.result_case' % (args.dataset, args.model_name)
        ensureDir(save_path_split)
        f_split = open(save_path_split, 'a')
        f_split.write(
            'lr=%.4f, norm=%.1f, gat=%d, lightgcn_iu=%d, hgcn=%d, hgcn_ug_side=%d, hgcn_u_hyperedge=%d, user_hpedge_ig=%d, contrastive_learning=%d, ssl_reg=%.7f, ssl_temp=%.2f, reweight_type=%d, beta_group=%.2f, beta_item=%.2f,regs=%s\n\t%s\n'
            % (args.lr, args.norm_2, args.gat, args.lightgcn_iu, args.hgcn, args.hgcn_ug_side,
               args.hgcn_u_hyperedge,
               args.user_hpedge_ig, args.contrastive_learning, args.ssl_reg, args.ssl_temp, args.reweight_type,
               args.beta_group, args.beta_item,
               args.regs, final_perf))
        f_split.close()


if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    args = parse_args()
    print(args)
    main(args)
