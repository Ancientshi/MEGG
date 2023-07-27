import argparse
import math

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def print_args(args):
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)

parser = argparse.ArgumentParser(description='LossChange for IRS')

#Path related
parser.add_argument('--data-path', default='''/home/yunxshi/Data/ml-20m''', type=str, help='The absolute path of the dataset.')
parser.add_argument('--root-path', default='''/home/yunxshi/Data/workspace/MEGG''', type=str, help='The absolute path of this project.')

#Inc parameters
parser.add_argument('--BLOCKNUM', default=15, type=int, help='Split dataset into blocks, blocknum is the number of bloks.')
parser.add_argument('--BASEBLOCK', default=10, type=int, help='Baseblock is the number of blocks for basic train.')
parser.add_argument('--method', default='full_batch', type=str, help='Choose in full_batch, fine_tune, losschange_replay')
parser.add_argument('--strategy', default='remain_sides', type=str, help='Sample strategy.')

#Hyper parameters
parser.add_argument('--replay_ratio', default=-1, type=float, help='Replaying Ration.')
parser.add_argument('--seed', default=0, type=int, help='Random seed.')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate.')
parser.add_argument('--weight-decay', default=0.01, type=float, help='Weight decay.')
parser.add_argument('--evaluation_nocold', action="store_true", help='Remove cold start users and items when evaluating.')
parser.add_argument('--gradient-fp', action="store_true", help='Calculate gradient with respect to full model parameters.')
parser.add_argument('--eva-future-one', action="store_true", help='Evaluate on future one incremental data block.')
parser.add_argument('--train-batch-size', default=1024, type=int, help='Train batch size.')
parser.add_argument('--losschange-batch-size', default=10000, type=int, help='Train batch size.')
parser.add_argument('--test-batch-size', default=4096, type=int, help='Test batch size.')
parser.add_argument('--embedding-dim', default=128, type=int, help='Embedding dim.')
parser.add_argument('--epoch', default=5, type=int, help='Train epoch.')

#Basic setting
parser.add_argument('--dataset', default='''ml-1m''', type=str, help='Choose in ml-1m, ml-20m, taobao2014, douban, lastfm-1k.')
parser.add_argument('--sample-size', default='''original''', type=str, help='500w, 30months, original.')
parser.add_argument('--task', default='''regression''', type=str, help='Choose in regression, binary.')
parser.add_argument('--net', default='''WDL''', type=str, help='Choose in WDL, DeepFM.')
parser.add_argument('--device', default=0, type=int, help='GPU device.')


args = parser.parse_args()
print_args(args)