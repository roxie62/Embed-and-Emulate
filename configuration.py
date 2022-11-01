import os, json, time, argparse, pdb, random, torch
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import AverageMeter, ProgressMeter,
from train_utils import calculate_parameter_error, compute_moment_score
from metricL_test import model_moment_func
from metricL_utils import MetricNet
from load_net import load_metric_net
# from multi_level_net import MultilevelNet

parser = argparse.ArgumentParser(description='End-to-End emulator')
# parameters for the dataset Loernz 96
parser.add_argument('--param_dim', default = 4, type = int)
parser.add_argument('--gt_param', default=[10., 1., 10., 10.], nargs='*', type = float)
parser.add_argument('--init_prior_mean', default=[10., 0., 2, 5.], nargs='*', type = float)
parser.add_argument('--init_prior_var', default=[10., 1., 0.1, 10.], nargs='*', type = float)
parser.add_argument('--sample_prior_min', default=[-2, -3, 0.1, 0], nargs='*', type = float)
parser.add_argument('--sample_prior_max', default=[20, 3, 20, 20], nargs='*', type = float)
parser.add_argument('--K', default = 36, type = int)
parser.add_argument('--J', default = 10, type = int)
parser.add_argument('--gt_intergrate_dt', default = 1000, type = int)
# parameters for the dataset KS
parser.add_argument('--kse', action = 'store_true')
# for evaluation
parser.add_argument('--noise_alpha', default = 0.1, type = float)

parser.add_argument('--use_bn_embed', action = 'store_true')
parser.add_argument('--use_adamw', action = 'store_true')
parser.add_argument('--seed', default = 1, type = int)
parser.add_argument('--gpu', default = 0, type = int)
parser.add_argument('--T', default = 100, type = int)

parser.add_argument('--total_epochs', default = 1000, type = int)
parser.add_argument('--train_size', default = 10000, type = int)
parser.add_argument('--batch_size_metricL', default = 250, type = int)
parser.add_argument('--bank_size', default = 0, type = int)

parser.add_argument('--dt', default = 0.1, type = float)

# for optimizer
parser.add_argument('--weight_decay', default = 1e-4, type = float)
parser.add_argument('--lr_ori', default = 1e-2, type = float)
parser.add_argument('--lr', default = 0.0001, type = float)

# Loss weights
parser.add_argument('--mape_traj_pri', default = 1, type = float)
parser.add_argument('--loss_param_traj_alone', default = 0.5, type = float)
parser.add_argument('--loss_param_alone_coeff', default = 1, type = float)
parser.add_argument('--loss_traj_alone_coeff', default = 1, type = float)
parser.add_argument('--loss_fix_param', default = 1, type = float)
# temperature value for contrastive and clip losses
parser.add_argument('--T_metricL', default = 0.1, type = float)
parser.add_argument('--T_metricL_param_alone', default = 0.1, type = float)
parser.add_argument('--T_metricL_traj_alone', default = 0.1, type = float)
parser.add_argument('--max_tau_metricL', default = 0.5, type = float)

parser.add_argument('--not_account_time', default = 1, type = int)
parser.add_argument('--alter_tau', action = 'store_true')
parser.add_argument('--pad', action = 'store_true')

# parser.add_argument('--Eval', action = 'store_true')
# parser.add_argument('--Eval_emulator', action = 'store_true')
# parser.add_argument('--Eval_metric', action = 'store_true')

parser.add_argument('--strict_filter', action = 'store_true')
parser.add_argument('--dist_index', default = 0, type = int) ## 0: AE; 1: MAPE: 2: NSE

parser.add_argument('--augpos_scale', default = 0.5, type = float)
parser.add_argument('--augpos_prob', default = 0.75, type = float)
parser.add_argument('--augpos_threshold', default = 0.45, type = float)

parser.add_argument('--load_saved_metric', action = 'store_true')

parser.add_argument('--run_oracle', action = 'store_true')
parser.add_argument('--npy_data_path', default = 'data_folder', type = str)
parser.add_argument('--nonlinear_data_path', default = '/net/scratch/roxie62/emulator/', type = str)
parser.add_argument('--data_path', default = '/net/scratch/roxie62/emulator/', type = str)
parser.add_argument('--extra_prefix', default = '', type = str)
parser.add_argument('--trainingdata_path', default = '', type = str)
## 0: sigma_star  1: sigma_identity
parser.add_argument('--sigma', default = 0, type = int)
parser.add_argument('--gt_idx', default = 0, type = int)
# embed network from here
parser.add_argument('--embed_dim', default = 64, type = int)
parser.add_argument('--hidden_dim_proj', default = 512, type = int)
parser.add_argument('--hidden_dim_param', default = 512, type = int)
parser.add_argument('--crop_T', default = 500, type = int)

# for distributed training
parser.add_argument('--distributed', action = 'store_true')
parser.add_argument('--multiprocessing_distributed', action = 'store_true')
parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--setting', default='0_0_0', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

args = parser.parse_args()
