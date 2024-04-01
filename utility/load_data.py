import numpy as np
import random as rd
import dgl

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train_1.txt'
        test_file = path + '/test_1.txt'
        ui_file = path + '/user_item.txt'

        #get number of users and items
        self.n_users, self.n_groups, self.n_items = 0, 0, 0
        self.n_train, self.n_test, self.ui_interactions = 0, 0, 0
        self.exist_users = []

        user_group_src = []
        user_group_dst = []
        user_item_src = []
        user_item_dst = []

        self.train_groups, self.test_set = {}, {}
        with open(train_file, 'r') as f:
            line = f.readline().strip()
            while line!= '':
                line = line.strip('\n').split(' ')
                uid = int(line[0])
                self.exist_users.append(uid)
                groups = [int(g) for g in line[1:]]
                self.train_groups[uid] = groups
                self.n_users = max(self.n_users, uid)
                self.n_groups = max(self.n_groups, max(groups))
                self.n_train += len(groups)
                for g in line[1:]:
                    user_group_src.append(uid)
                    user_group_dst.append(int(g))
                line = f.readline().strip()

        with open(test_file, 'r') as f:
            line = f.readline().strip()
            while line!= '':
                line = line.strip('\n').split(' ')
                uid = int(line[0])
                groups_test = [int(g) for g in line[1:]]
                self.test_set[uid] = groups_test
                self.n_groups = max(self.n_groups, max(groups_test))
                self.n_test += len(groups_test)
                line = f.readline().strip()
        self.n_users += 1
        self.n_groups += 1

        with open(ui_file, 'r') as f:
            line = f.readline().strip()
            while line!= '':
                line = line.strip('\n').split(' ')
                uid = int(line[0])
                items = [int(i) for i in line[1:]]
                self.n_items = max(self.n_items, max(items))
                self.ui_interactions += len(items)
                for i in line[1:]:
                    user_item_src.append(uid)
                    user_item_dst.append(int(i))
                line = f.readline().strip()
        self.n_items += 1
        
        self.print_statistics()

        # self.train_groups, self.test_set = {}, {}
        # with open(train_file) as f_train:
        #     with open(test_file) as f_test:
        #         for l in f_train.readlines():
        #             if len(l) == 0:
        #                 break
        #             l = l.strip('\n')
        #             groups = [int(i) for i in l.split(' ')]
        #             uid, train_groups = groups[0], groups[1:]
        #             self.train_groups[uid] = train_groups
        #
        #         for l in f_test.readlines():
        #             if len(l) == 0: break
        #             l = l.strip('\n')
        #             try:
        #                 groups = [int(i) for i in l.split(' ')]
        #             except Exception:
        #                 continue
        #
        #             uid, test_groups = groups[0], groups[1:]
        #             self.test_set[uid] = test_groups
        
        data_dict = {
            ('user', 'ug', 'group'): (user_group_src, user_group_dst),
            ('group', 'gu', 'user'): (user_group_dst, user_group_src),
            ('user', 'ui', 'item'): (user_item_src, user_item_dst),
            ('item', 'iu', 'user'): (user_item_dst, user_item_src)
        }
        num_dict = {
            'user': self.n_users, 'group': self.n_groups, 'item': self.n_items
        }
        self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)
        self.user_group_src = user_group_src
        self.user_group_dst = user_group_dst
        self.user_item_src = user_item_src
        self.user_item_dst = user_item_dst

    def print_statistics(self):
        print('n_users=%d, n_groups=%d, n_items=%d' % (self.n_users, self.n_groups, self.n_items))
        print('n_ug_interactions=%d, n_ui_interactions=%d' % (self.n_train + self.n_test, self.ui_interactions))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
        self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_groups)))

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_groups_for_u(u, num):
            # sample num pos items for u-th user
            pos_groups = self.train_groups[u]
            n_pos_groups = len(pos_groups)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_groups, size=1)[0]
                pos_i_id = pos_groups[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_groups_for_u(u, num):
            # sample num neg items for u-th user
            neg_groups = []
            while True:
                if len(neg_groups) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_groups, size=1)[0]
                if neg_id not in self.train_groups[u] and neg_id not in neg_groups:
                    neg_groups.append(neg_id)
            return neg_groups

        pos_groups, neg_groups = [], []
        for u in users:
            pos_groups += sample_pos_groups_for_u(u, 1)
            neg_groups += sample_neg_groups_for_u(u, 1)

        return users, pos_groups, neg_groups




