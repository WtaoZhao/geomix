from models import *

def parse_method(args, c, d, device):
    if args.encoder=='geomix_1':
        model=GeoMix1(d,
            args.hidden_channels,
            c,
            args.num_layers,
            args.dropout,
            args.use_bn,
            args.alpha,
            args.hops).to(device)
    elif args.encoder=='geomix_2':
        model=GeoMix2(
            d,
            args.hidden_channels,
            c,
            args.num_layers,
            args.hops,
            args.alpha,
            args.dropout,
            args.use_bn).to(device)
    elif args.encoder=='geomix_3':
        model=GeoMix3(
            d,
            args.hidden_channels,
            c,
            args.num_layers,
            args.dropout,
            args.use_bn,
            args.res_weight,
            args.hops,
            args.graph_weight,
            args.use_weight,
            args.attn_emb_dim
            ).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=300)

    # model network
    parser.add_argument('--method', type=str, default='augTrain')
    parser.add_argument('--encoder', type=str, default='geomix_1')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for deep methods')
    parser.add_argument('--hops', type=int, default=1, help='power of adjacency matrix for certain methods')

    # geomix_3
    parser.add_argument('--res_weight', type=float, default=0.5, help='residual weight. 0 means not use residual.')
    parser.add_argument('--graph_weight', type=float, default=0.8, help='graph weight in augmentation. 0 means not use graph.')
    parser.add_argument('--use_weight', action='store_true', help='use weight matrix to transform query and key')
    parser.add_argument('--attn_emb_dim', type=int, default=16, help='dimension of attention weight matrix')

    #augTrain
    parser.add_argument('--label_grad', action='store_true', help='take gradients w.r.t pseudolabels')
    parser.add_argument('--conf_thresh', type=float, default=0,help='confidence threshold')
    parser.add_argument('--aug_lamb', type=float, default=1,help='Augmentation loss weight.')
    parser.add_argument('--alpha', type=float, default=0,help='Skip weight in APP_GCN augmentation.')

    # training
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')
    parser.add_argument('--no_feat_norm', action='store_true', help='not use feature normalization')

    # image data
    parser.add_argument('--label_num_per_class', type=int, default=50, help='number of training samples per class')
    parser.add_argument('--valid_num', type=int, default=5000, help='number of validation samples')


    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=50, help='how often to print')
    parser.add_argument('--output_folder', type=str,
                        default='results', help='Output folder')
