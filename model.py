import sys

import torch
import torch.nn as nn
import numpy as np
import sklearn
import torch.nn.functional as F
import dgl
from utility.LightGCNLayer import LightGCNLayer
from utility.AttLayer import MultiHeadGATLayer
import dgl.function as fn
import torchmetrics

def construct_user_group_bigraph(graph):
    return graph.node_type_subgraph(['user', 'group'])


def construct_negative_graph(graph, k, device):
    user_group_src, user_group_dst = graph.edges(etype='ug')
    neg_src = user_group_src.repeat_interleave(k)
    n_neg_src = len(user_group_src)
    neg_dst = torch.randint(0, graph.num_nodes(ntype='group'), (n_neg_src * k,)).to(device)
    # print(torch.count_nonzero(neg_dst - neg_src) == neg_dst.shape[0])
    # while torch.all(neg_dst - neg_src) != True:
    #     neg_dst = torch.randint(0, graph.num_nodes(ntype='group'), (n_neg_src * k,)).to(device)
    # print(torch.count_nonzero(neg_dst - neg_src) == neg_dst.shape[0])
    # sys.exit(0)
    data_dict = {
        ('user', 'ug', 'group'): (neg_src, neg_dst),
        ('group', 'gu', 'user'): (neg_dst, neg_src),
        # ('neg_user', 'ui', 'item'): (user_item_src, user_item_dst),
        # ('item', 'iu', 'neg_user'): (user_item_dst, user_item_src)
    }
    num_dict = {
        'user': graph.num_nodes(ntype='user'), 'group': graph.num_nodes(ntype='group'),
        # 'item': graph.num_nodes(ntype='item')
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']




# class HGCNLayer(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, graph, h, etype_forward, etype_back, norm_2=-1):
#         with graph.local_scope():
#             src, _, dst = etype_forward
#             feat_src = h[src]
#             feat_dst = h[dst]
#             aggregate_fn = fn.copy_src('h', 'm')
#             aggregate_fn_back = fn.copy_src('h_b', 'm_b')
#
#             graph.nodes[src].data['h'] = feat_src
#             graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype_forward)
#
#             rst = graph.nodes[dst].data['h']
#             in_degrees = graph.in_degrees(etype=etype_forward).float().clamp(min=1)
#             norm_dst = torch.pow(in_degrees, -1)
#             shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
#             norm = torch.reshape(norm_dst, shp_dst)
#             rst = rst * norm
#
#             graph.nodes[dst].data['h_b'] = rst
#             graph.update_all(aggregate_fn_back, fn.sum(msg='m_b', out='h_b'), etype=etype_back)
#             bsrc = graph.nodes[src].data['h_b']
#
#             in_degrees_b = graph.in_degrees(etype=etype_back).float().clamp(min=1)
#             norm_src = torch.pow(in_degrees_b, norm_2)
#             shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
#             norm_src = torch.reshape(norm_src, shp_src)
#             bsrc = bsrc * norm_src
#
#             return bsrc, rst

class HGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype_forward, etype_back, norm_2=-1, alpha = 0):
        with graph.local_scope():
            src, _, dst = etype_forward
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_src('h', 'm')
            aggregate_fn_back = fn.copy_src('h_b', 'm_b')

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype_forward)

            rst = graph.nodes[dst].data['h']
            in_degrees = graph.in_degrees(etype=etype_forward).float().clamp(min=1)
            norm_dst = torch.pow(in_degrees, -1)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm_dst, shp_dst)
            # rst = rst * norm
            if alpha != 0:
                # print(rst)
                # print(feat_dst)
                rst = rst + alpha * feat_dst
                # print(rst)
                # sys.exit(0)

            rst = rst * norm
            graph.nodes[dst].data['h_b'] = rst
            graph.update_all(aggregate_fn_back, fn.sum(msg='m_b', out='h_b'), etype=etype_back)
            bsrc = graph.nodes[src].data['h_b']

            in_degrees_b = graph.in_degrees(etype=etype_back).float().clamp(min=1)
            norm_src = torch.pow(in_degrees_b, norm_2)
            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            norm_src = torch.reshape(norm_src, shp_src)
            bsrc = bsrc * norm_src

            return bsrc, rst

class HGCNLayer_general(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype_list, norm_2=-1):
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')
            aggregate_fn_back = fn.copy_src('h_b', 'm_b')
            for etype in etype_list:
                etype_forward, _ = etype
                src, _, dst = etype_forward
                feat_src = h[src]
                feat_dst = h[dst]

                graph.nodes[src].data['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype_forward)

                rst = graph.nodes[dst].data['h']
                in_degrees = graph.in_degrees(etype=etype_forward).float().clamp(min=1)
                norm_dst = torch.pow(in_degrees, -1)
                shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm_dst, shp_dst)
                rst = rst * norm
                graph.nodes[dst].data['h_b'] = rst

            update_dict = {}
            in_degrees_b = None
            for etype in etype_list:
                _, etype_back = etype
                update_dict[etype_back] = (aggregate_fn_back, fn.sum(msg='m_b', out='h_b'))
                if in_degrees_b == None:
                    in_degrees_b = graph.in_degrees(etype=etype_back).float().clamp(min=1)
                else:
                    in_degrees_b += graph.in_degrees(etype=etype_back).float().clamp(min=1)
            graph.multi_update_all(update_dict, 'sum')
            bsrc = graph.nodes[src].data['h_b']

            norm_src = torch.pow(in_degrees_b, norm_2)
            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            norm_src = torch.reshape(norm_src, shp_src)
            bsrc = bsrc * norm_src

            return bsrc, rst


class GAT(nn.Module):
    def __init__(self, args, graph, in_dim, hidden_dim, out_dim, num_heads=2):
        super(GAT, self).__init__()
        print(in_dim,hidden_dim,out_dim)
        self.gat = args.gat
        self.hid_dim = in_dim
        self.neg_samples = args.neg_samples
        self.decay = eval(args.regs)[0]
        self.user_embedding = torch.nn.Parameter(torch.randn(graph.nodes('user').shape[0], self.hid_dim))
        self.group_embedding = torch.nn.Parameter(torch.randn(graph.nodes('group').shape[0], self.hid_dim))
        self.item_embedding = torch.nn.Parameter(torch.randn(graph.nodes('item').shape[0], self.hid_dim))


        if self.gat == 2:
            self.layer0_iu = MultiHeadGATLayer(in_dim, out_dim, num_heads)
            # self.layer0_ui = MultiHeadGATLayer(in_dim, out_dim, num_heads)
        self.layer1_ug = MultiHeadGATLayer(in_dim, out_dim, num_heads)
        # self.layer1_gu = MultiHeadGATLayer(in_dim, out_dim, num_heads)

        torch.nn.init.normal_(self.user_embedding, std=0.1)
        torch.nn.init.normal_(self.group_embedding, std=0.1)
        self.node_features = {'user': self.user_embedding, 'group': self.group_embedding, 'item': self.item_embedding}
        self.pred = ScorePredictor()
    def forward(self, graph):
        h = self.node_features
        if self.gat == 2:
            user_embed = self.layer0_iu(graph, h, ('item', 'iu', 'user'))
            user_embed = torch.nn.functional.elu(user_embed)
            h = {'user': user_embed, 'group': h['group']}
        # user_embed = self.layer1_gu(graph, h, ('group', 'gu', 'user'))
        # user_embed = torch.nn.functional.elu(user_embed)
        group_embed = self.layer1_ug(graph, h, ('user', 'ug', 'group'))
        group_embed = torch.nn.functional.elu(group_embed)
        h = {'user': h['user'], 'group': group_embed}
        return h

    def create_bpr_loss(self, pos_g, neg_g, h, g):
        pos_score = self.pred(pos_g, h)
        neg_score = self.pred(neg_g, h)
        pos_score = pos_score[('user', 'ug', 'group')].repeat_interleave(self.neg_samples, dim=0)
        neg_score = neg_score[('user', 'ug', 'group')]
        # reweight_norm_group = torch.unsqueeze((1 - beta_group) / (1 - beta_group ** reweight_norm_group), 1)
        mf_loss = nn.Softplus()(neg_score - pos_score)
        mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (self.user_embedding.norm(2).pow(2) +
                                 self.group_embedding.norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss

class LightGCN(nn.Module):
    def __init__(self, args, graph, device):
        super().__init__()
        self.hid_dim = args.embed_size
        self.layer_num = args.layer_num
        self.neg_samples = args.neg_samples
        self.decay = eval(args.regs)[0]
        self.n_user = graph.nodes('user').shape[0]
        self.lightgcn_iu = args.lightgcn_iu

        self.pre_gcn = args.pre_gcn
        self.hgcn_mix = args.hgcn_mix
        self.hgcn = args.hgcn
        self.hgcn_ug_side = args.hgcn_ug_side
        self.hgcn_u_hyperedge = args.hgcn_u_hyperedge
        self.hgcn_g_hyperedge = self.hgcn_ug_side - self.hgcn_u_hyperedge
        self.norm_2 = args.norm_2
        self.reweight_type = args.reweight_type
        self.beta_group = args.beta_group
        self.beta_item = args.beta_item
        self.user_hpedge_ig = args.user_hpedge_ig

        self.contrastive_learning = args.contrastive_learning

        self.user_embedding = torch.nn.Parameter(torch.randn(graph.nodes('user').shape[0], self.hid_dim))
        self.user_hyperedge = torch.empty((graph.nodes('user').shape[0], self.hid_dim), requires_grad=False)

        self.group_embedding = torch.nn.Parameter(torch.randn(graph.nodes('group').shape[0], self.hid_dim))
        self.group_hyperedge = torch.empty((graph.nodes('group').shape[0], self.hid_dim), requires_grad=False)

        self.item_hyperedge = torch.empty((graph.nodes('item').shape[0], self.hid_dim), requires_grad=False)

        torch.nn.init.normal_(self.user_embedding, std=0.1)
        torch.nn.init.normal_(self.group_embedding, std=0.1)
        self.build_model()
        if self.pre_gcn == 2 or self.lightgcn_iu or self.user_hpedge_ig == 4:
            self.item_embedding = torch.nn.Parameter(torch.randn(graph.nodes('item').shape[0], self.hid_dim))
            torch.nn.init.normal_(self.item_embedding, std=0.1)
            self.node_features = {'user': self.user_embedding, 'group': self.group_embedding, 'item': self.item_embedding}
        else:
            self.node_features = {'user': self.user_embedding, 'group': self.group_embedding, 'item': self.item_hyperedge}

        self.pred = ScorePredictor()

    def build_layer(self, idx=0):
        return LightGCNLayer()



    def build_model(self):
        self.HGCNlayer = HGCNLayer()
        self.HGCNlayer_general = HGCNLayer_general()
        self.layers = nn.ModuleList()
        self.LGCNlayer = LightGCNLayer()
        for idx in range(self.layer_num):
            h2h = self.build_layer(idx)
            self.layers.append(h2h)

    def forward(self, graph):
        h = self.node_features
        user_embed = [self.user_embedding]
        group_embed = [self.group_embedding]
        norm = self.norm_2
        flag_lightgcn = 1
        if self.contrastive_learning:
            if self.contrastive_learning!= -1:
                self.h_user_v2, _ = self.HGCNlayer(graph, h, ('user', 'ug', 'group'), ('group', 'gu', 'user'), norm)
            if self.contrastive_learning in [-1,2]:
                self.h_group_v2, _ = self.HGCNlayer(graph, h, ('group', 'gu', 'user'),('user', 'ug', 'group'), norm)
        if self.hgcn:
            if self.user_hpedge_ig == 1:
                h_user, _ = self.HGCNlayer_general(graph, h, [(('user', 'ui', 'item'), ('item', 'iu', 'user')),
                                                              (('user', 'ug', 'group'), ('group', 'gu', 'user'))], norm)
            elif self.user_hpedge_ig == 2:
                if self.pre_gcn == 2:
                    h['item'] = self.item_embedding
                    h_user = self.LGCNlayer(graph, h, ('item', 'iu', 'user'), norm)
                else:
                    h_user, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), norm)
                h = {'user': h_user, 'group': self.group_embedding}
                h_user, _ = self.HGCNlayer(graph, h, ('user', 'ug', 'group'), ('group', 'gu', 'user'), norm)
            elif self.user_hpedge_ig == 3:
                h_user, _ = self.HGCNlayer(graph, h, ('user', 'ug', 'group'), ('group', 'gu', 'user'), norm)
            elif self.user_hpedge_ig == 4:
                h_item, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'),norm)
                h = {'user': self.user_embedding, 'group': self.group_embedding,'item':h_item}
                h_user, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), norm)
                h = {'user': h_user, 'group': self.group_embedding}
                h_user, _ = self.HGCNlayer(graph, h, ('user', 'ug', 'group'), ('group', 'gu', 'user'), norm)
            else:
                h_user, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), norm)
            if self.contrastive_learning:
                self.h_user_v1 = h_user
            h = {'user': h_user, 'group': self.group_embedding}
            if self.hgcn_ug_side:
                flag_lightgcn = 0
                if self.hgcn_u_hyperedge:
                    if self.pre_gcn == 1:
                        h_group = self.LGCNlayer(graph, h, ('user', 'ug', 'group'),norm) + self.group_embedding
                        h = {'user': h_user, 'group': h_group}
                    # if self.contrastive_learning == 2:
                    #     self.h_group_v2 = self.LGCNlayer(graph, h, ('user', 'ug', 'group'),norm)
                    h_group, _ = self.HGCNlayer(graph, h, ('group', 'gu', 'user'), ('user', 'ug', 'group'), norm, self.hgcn_mix)
                    if self.contrastive_learning in [-1,2]:
                        self.h_group_v1 = h_group
                    #     print(self.h_group_v1)
                    #     print(self.h_group_v2)
                    #     sys.exit(0)
                    h = {'user': h_user, 'group': h_group}
                elif self.hgcn_g_hyperedge:
                    '''group as hyperedge without gradient'''
                    h_user, h_group = self.HGCNlayer(graph, h, ('user', 'ug', 'group'), ('group', 'gu', 'user'))
                    h = {'user': h_user, 'group': h_group.detach()}
        if flag_lightgcn != 0 and self.layer_num != 0:
            for layer in self.layers:
                if self.lightgcn_iu:
                    h_item = layer(graph, h, ('user', 'ui', 'item'),norm)
                    h_user = layer(graph, h, ('item', 'iu', 'user'),norm)
                    h = {'user': h_user, 'item': h_item, 'group':h['group']}
                h_group = layer(graph, h, ('user', 'ug', 'group'),norm)
                h_user = layer(graph, h, ('group', 'gu', 'user'),norm)
                if self.lightgcn_iu:
                    h = {'user': h_user, 'group': h_group, 'item': h['item']}
                else:
                    h = {'user': h_user, 'group': h_group,}
                user_embed.append(h_user)
                group_embed.append(h_group)
            user_embed = torch.mean(torch.stack(user_embed, dim=0), dim=0)
            group_embed = torch.mean(torch.stack(group_embed, dim=0), dim=0)
            h = {'user': user_embed, 'group': group_embed}
        return h

    def alignment_score(self):
        h_user_v1 = F.normalize(self.h_user_v1,p=2,dim=-1)
        h_user_v2 = F.normalize(self.h_user_v2,p=2,dim=-1)
        # pdist = nn.PairwiseDistance(p=2)
        # return pdist(h_user_v1, h_user_v2).mean()
        cosine_similarity = torchmetrics.CosineSimilarity(reduction=None)
        # return sklearn.metrics.pairwise.cosine_similarity(h_group_v1[idx]).mean()
        result = cosine_similarity(h_user_v1,h_user_v2).cpu().detach().numpy()
        return np.nanmean(result)

    def uniformity_score_user(self):
        h_user_v1 = F.normalize(self.h_user_v1,p=2,dim=-1)
        # return torch.pdist(h_user_v1,p=2).mean()
        result = torchmetrics.functional.pairwise_cosine_similarity(h_user_v1).cpu().detach().numpy()
        return np.nanmean(result)

    def uniformity_score_group(self):
        h_group_v1 = F.normalize(self.h_group_v1,p=2,dim=-1)
        return torchmetrics.functional.pairwise_cosine_similarity(h_group_v1).mean()
        # return torch.pdist(h_group_v1,p=2).mean()

    def uniformity_score_group_batch(self):
        output = 0
        k = 4
        h_group_v1 = np.array_split(F.normalize(self.h_group_v1,p=2,dim=-1).cpu().detach().numpy(),k)
        for idx in range(k):
            # output += torch.pdist(x,p=2).mean()
            # print(x.shape)
            # print(torchmetrics.functional.pairwise_cosine_similarity(x))
            # print(h_group_v1[idx].shape)
            # print(sklearn.metrics.pairwise.cosine_similarity(h_group_v1[idx]).shape)
            output += sklearn.metrics.pairwise.cosine_similarity(h_group_v1[idx]).mean()
        return output/k

    def create_ssl_loss_user(self, ssl_temp):
        # ssl_temp = 0.1
        h_user_v1 = torch.nn.functional.normalize(self.h_user_v1, p=2, dim=1)
        h_user_v2 = torch.nn.functional.normalize(self.h_user_v2, p=2, dim=1)
        pos_score = torch.sum(torch.mul(h_user_v1, h_user_v2), dim=1)
        neg_score = torch.matmul(h_user_v1, h_user_v2.T)
        pos_score = torch.exp(pos_score / ssl_temp)
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def create_ssl_loss_group(self, ssl_temp):
        ssl_temp = 0.1
        h_group_v1 = torch.nn.functional.normalize(self.h_group_v1, p=2, dim=1)
        h_group_v2 = torch.nn.functional.normalize(self.h_group_v2, p=2, dim=1)
        # pos_score = torch.sum(torch.mul(h_group_v1, h_group_v2), dim=1)
        neg_score = torch.matmul(h_group_v1, h_group_v2.T)
        pos_score = 1/ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss  = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def create_ssl_loss_batched_group(self, ssl_temp, k = 4, idx = 0):
        ssl_temp = 0.1
        h_group_v1 = self.h_group_v1.split(self.h_group_v1.shape[0] // k + 1)[idx]
        h_group_v2 = self.h_group_v2.split(self.h_group_v2.shape[0] // k + 1)[idx]
        h_group_v1 = torch.nn.functional.normalize(h_group_v1, p=2, dim=1)
        h_group_v2 = torch.nn.functional.normalize(h_group_v2, p=2, dim=1)
        # pos_score = torch.sum(torch.mul(h_group_v1, h_group_v2), dim=1)
        neg_score = torch.matmul(h_group_v1, h_group_v2.T)
        pos_score = 1 / ssl_temp
        neg_score = torch.sum(torch.exp(neg_score / ssl_temp), axis=1)
        ssl_loss  = -torch.sum(torch.log(pos_score / neg_score))
        return ssl_loss

    def reweight_formula(self, beta, x, type=0):
        if type == 0:
            fx = (1 - beta) / (1 - beta ** x)
        elif type == 1:
            fx = 1 / (beta * x - beta + 1)
        elif type == 2:
            fx = 1 / torch.exp(beta * x - beta)
        elif type == 3:
            fx = 1 - torch.tanh(beta * x - beta)
        return torch.unsqueeze(fx, 1)

    def create_bpr_loss(self, pos_g, neg_g, h, g):
        # print(h['user'])
        # print(h['group'])
        pos_score = self.pred(pos_g, h)
        neg_score = self.pred(neg_g, h)
        beta_group = self.beta_group
        beta_item = self.beta_item
        pos_score = pos_score[('user', 'ug', 'group')].repeat_interleave(self.neg_samples, dim=0)
        neg_score = neg_score[('user', 'ug', 'group')]
        '''group-side loss reweight'''
        if beta_group !=0:
            user_in_degrees_group = pos_g.out_degrees(etype=('user', 'ug', 'group'))
            reweight_norm_group = user_in_degrees_group.repeat_interleave(user_in_degrees_group, dim=0)
            reweight_norm_group = self.reweight_formula(beta_group, reweight_norm_group, self.reweight_type)
        '''item-side loss reweight'''
        if beta_item != 0:
            user_in_degrees_item = g.out_degrees(etype=('user', 'ui', 'item')).clamp(min=1)
            reweight_norm_item = user_in_degrees_item.repeat_interleave(user_in_degrees_group, dim=0)
            reweight_norm_item = self.reweight_formula(beta_item, reweight_norm_item, self.reweight_type)
        mf_loss = nn.Softplus()(neg_score - pos_score)
        if beta_group !=0 or beta_item !=0:
            mf_loss = mf_loss * reweight_norm_group * reweight_norm_item
        mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (self.user_embedding.norm(2).pow(2) +
                                 self.group_embedding.norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss
