import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--data_path', nargs='?', default='./Data/',
                        help='Input data path.')
    parser.add_argument('--weights_path', nargs='?', default='./Weights/',
                        help='Input data path.')
    parser.add_argument('--model_name', type=str, default='crossvalidation',
                        help='Saved model name.')
    parser.add_argument('--dataset', nargs='?', default='beibei')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=10000,
                        help='Number of epoch.')
    parser.add_argument('--user_split', type=int, default=0)
    parser.add_argument('--gat', type=int, default=0)
    parser.add_argument('--lightgcn_iu', type=int, default=1)
    parser.add_argument('--hgcn', type=int, default=0)
    parser.add_argument('--pre_gcn', type=int, default=0)
    parser.add_argument('--combine_view', type=int, default=0)
    parser.add_argument('--show_distance', type=int, default=0)
    parser.add_argument('--combine_alpha', type=float, default=0.2)
    parser.add_argument('--hgcn_mix', type=float, default=0)

    parser.add_argument('--hgcn_ug_side', type=int, default=1,
                        help='Hypergraph conv on user-group side instead of LightGCN')
    parser.add_argument('--hgcn_u_hyperedge', type=int, default=1,
                        help='Hypergraph conv on user-group side with user as hyperedge and group as vertex')

    parser.add_argument('--norm_2', type=int, default=-1,
                        help='-0.5 for steam/mafengwo, -1 for beibei/weeplaces')

    parser.add_argument('--user_hpedge_ig', type=int, default=0,
                        help='Hypergraph conv on user side with user as vertex'
                             '{0: only user-item conv with item as hyperedge;'
                             ' 1: simultaneous user-item and user-group conv;'
                             ' 2: sequential user-item and user-group conv};'
                             '3&4:HGNN and HGNN+')

    parser.add_argument('--contrastive_learning', type=int, default=0,
                        help='0: no CL; 1: only user; 2: both user and group; -1: only group')

    parser.add_argument('--ssl_reg', type=float, default=1e-7,
                        help='1e-5 for steam')
    parser.add_argument('--ssl_temp', type=float, default=0.1,
                        help='temperature, 0.1 as default')

    parser.add_argument('--reweight_type', type=int, default=2,
                        help='0: (1-b)/(1-b^x); 1: 1/(bx-b+1); 2: 1/(e^(bx-b)); 3: 1-tanh(bx-b)')
    parser.add_argument('--beta_group', type=float, default=0)
    parser.add_argument('--beta_item', type=float, default=0)

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_num', type=int, default=3,
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-7]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate. 0.05/0.005 for steam(beibei)/weeplaces')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='Number of negative samples.')

    parser.add_argument('--flag_step', type=int, default=5,
                        help='Number of negative samples.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--Ks', nargs='?', default='[10,20]',
                        help='Output sizes of every layer')

    parser.add_argument('--fast_test', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    # parser.add_argument('--mtv_att', type=int, default=0,
    #                     help='0 for no att, 1 for mtv_att')
    # parser.add_argument('--att_type', type=int, default=2,
    #                     help='Att_type')
    # parser.add_argument('--att_update_user', type=int, default=3,
    #                     help='0 for no linear_transform, 1 for linear_transform')
    # parser.add_argument('--res_lambda', type=float, default=0.001,
    #                     help='res_lambda')
    # parser.add_argument('--linear_transform', type=int, default=1,
    #                     help='0 for no linear_transform, 1 for linear_transform')
    # parser.add_argument('--initial_embedding_used', type=int, default=1,
    #                     help='0 for no att, 1 for mtv_att')
    # parser.add_argument('--item_lambda', type=float, default=100,
    #                     help='item_lambda.')
    return parser.parse_args()
