from models import *


def parse_method(args, c, d, device):
    method=args.encoder
    if method=='gcn':
        model=GCN(d, args.hidden_channels, c, args.num_layers, args.dropout, use_bn=args.use_bn).to(device)
    else:
        raise ValueError(f'Invalid method {method}')
    return model


def parser_add_main_args(parser):
    # setup
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--data_dir', type=str,
                        default='../../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20,
                        help='labeled nodes per class(randomly selected)')
    parser.add_argument('--valid_num', type=int, default=500,
                        help='Total number of validation')
    parser.add_argument('--test_num', type=int, default=500,
                        help='Total number of test')

    # model
    parser.add_argument('--method', type=str, default='gcn')
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_residual', action='store_true',
                        help='use residual link for each GNN layer')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    
    # trainer
    parser.add_argument('--aug_mod', type=str, default='gcn')
    parser.add_argument('--label_grad', action='store_true', help='take gradients w.r.t pseudolabels')
    parser.add_argument('--aug_lamb', type=float, default=1,help='Augmentation loss weight.')
    parser.add_argument('--alpha', type=float, default=0.5,help='Skip weight in APP_GCN augmentation.')
    parser.add_argument('--res_weight', type=float, default=0.5, help='residual weight. 0 means not use residual.')
    parser.add_argument('--graph_weight', type=float, default=0.8, help='graph weight in augmentation. 0 means not use graph.')
    parser.add_argument('--use_weight', action='store_true', help='use weight matrix to transform query and key')
    parser.add_argument('--attn_emb_dim', type=int, default=16, help='dimension of attention weight matrix')

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')

    parser.add_argument('--no_feat_norm', action='store_true',
                        help='Not use feature normalization.')
    parser.add_argument('--patience', type=int, default=200,
                        help='early stopping patience.')
