filter_fn = filter
from configuration import *
from dataloader import *
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from metricL_utils import MetricNet, ParamEmbed
from metricL_utils_vit import MetricNet_ViT
from mpdb import mpdb
from utils import init_distributed_mode, HiddenPrints
import time, math, copy
from PIL import Image
import warnings
import builtins
from metricL_eval_torch import create_eval_eki_with_metric_model

def adjust_learning_rate_cos(lr_ori, optimizer, epoch, total_epochs, args, warm_up_epochs = 10):
    """Decay the learning rate based on schedule"""
    lr = lr_ori
    if epoch < warm_up_epochs:
        lr = lr_ori * (epoch + 1) / warm_up_epochs
    else:
        lr *= 0.5 * (1 + math.cos(math.pi * (epoch - warm_up_epochs) / total_epochs))
        # lr += 5e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_tau_metricL(T_metricL_ori, train_epoch, total_epochs, epochs = 100, max_tau_metricL = 0.5):
    if train_epoch <= (total_epochs - epochs):
        T_metricL = T_metricL_ori + (max_tau_metricL - T_metricL_ori) * max((train_epoch - int(total_epochs/2)), 0) / ((total_epochs-epochs) - int(total_epochs/2))
    else:
        T_metricL = max_tau_metricL
    return T_metricL

def main(args):
    print(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        #torch.set_deterministic(True)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size >= 1
    ngpus_per_node = torch.cuda.device_count()

    print('start')
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    args.gpu = args.gpu % torch.cuda.device_count()
    print('world_size', args.world_size)
    print('rank', args.rank)
    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    args.distributed = args.world_size >= 1 or args.multiprocessing_distributed
    main_worker(args.gpu, ngpus_per_node, args)


@torch.no_grad()
def run_eval(input_list, input_args, epoch):
    args = copy.deepcopy(input_args)
    args.distributed = False
    img_folder_eval  = args.img_folder_eval + '/result.txt'

    input_metric_model, input_param_model, _, _ = input_list
    metric_model, param_model, moment_func, batch_moment_func, _ = load_metric_net(args)
    metric_model.cuda(args.gpu)
    param_model.cuda(args.gpu)
    metric_model.load_state_dict(input_metric_model.module.state_dict())
    param_model.load_state_dict(input_param_model.module.state_dict())
#
    metric_model.eval()
    param_model.eval()
    moment_func, batch_moment_func = model_moment_func(metric_model)

    print('start eval at epoch {}'.format(epoch))
    #with HiddenPrints():
    with torch.no_grad():
        for args.add_noise_alpha in [0]:
            run_eki_func = create_eval_eki_with_metric_model(args, save_list = False)
            averaged_enki, median_enki, averaged_regression, median_regression = run_eki_func(metric_model, param_model, moment_func, batch_moment_func, args.img_folder_eval, args, idx = None)
            if args.gpu == 0:
                with open(img_folder_eval, 'a+') as f:
                    f.writelines('******epoch: {:05d}*********\n'.format(epoch))
                    f.writelines('averaged enki: {}\n'.format(averaged_enki))
                    f.writelines('median enki: {}\n'.format(median_enki))
                    f.writelines('averaged regression: {}\n'.format(averaged_regression))
                    f.writelines('median regression: {}\n'.format(median_regression))
                    f.writelines('*****************\n\n\n')

    if args.gpu == 0:
        print('done eval at epoch {}'.format(epoch))
        print('averaged enki: {}'.format(averaged_enki))
        print('median enki: {}'.format(median_enki))
        print('averaged regression: {}'.format(averaged_regression))
        print('median regression: {}'.format(median_regression))

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.is_master = args.rank % ngpus_per_node == 0 and args.gpu == 0


    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    img_name = None

    ## load model
    train_epoch_ = 0

    img_name = trainsz_{}_{}_memory_bank_{}'.format(\
                args.train_size, args.batch_size_metricL, args.train_size + args.bank_size)

    load_img_name = None
    load_flag = False
    if args.load_saved_metric:
        saved_pth_list = glob.glob('saved_checkpoint_path/{}metric_epch*'.format(img_name))

        saved_pth_list.sort()
        if len(saved_pth_list) > 0:
            epoch_id = np.array([int(p.split('_')[-1]) for p in saved_pth_list])
            #train_epoch_ = int(saved_pth_list[-1].split('/')[-1].split('_')[-1])
            train_epoch_ = epoch_id.max() + 1
            load_img_name = img_name
            load_flag = True
            print('load checkpoint: {} at {} epoch'.format(load_img_name, train_epoch_))

    metric_model, param_model, moment_func, batch_moment_func, optimizer = load_metric_net(args, img_name = load_img_name, \
                                                                            train_epoch = train_epoch_ - 1)

    lr_ori = args.lr_ori
    print('learning rate', lr_ori)

    ## for synchorization
    metric_model = nn.SyncBatchNorm.convert_sync_batchnorm(metric_model)
    param_model = nn.SyncBatchNorm.convert_sync_batchnorm(param_model)

    ## Parameters
    K, J = args.K, args.J
    crop_T = args.crop_T

    batch_size = args.batch_size_metricL
    train_dataset = TrainingData(args.crop_T, args, train_size = args.train_size)

    args.trainingdata_path = train_dataset.data_path
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle = True)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle = True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=3, pin_memory=True, sampler=train_sampler, drop_last=True, persistent_workers = True)


    gt_idx = args.gt_idx
    total_epochs = args.total_epochs
    use_mem_bank = False
    T_metricL_ori = args.T_metricL_traj_alone
    T_metricL_traj_alone = args.T_metricL_traj_alone
    T_metricL_param_alone = args.T_metricL_param_alone
    T_metricL = args.T_metricL
    no_fix_900 = False
    max_tau_metricL = args.max_tau_metricL
    alter_tau_metricL = args.alter_tau
    alter_tau_inter = args.alter_tau_inter
    from train_utils import calculate_parameter_loss
    # lambda_loss_oracle = args.loss_oracle

    loss_list_param, loss_list_traj = [], []
    pri_hat_param_mape = []
    param_alone_loss_list, traj_alone_loss_list = [], []
    idx_list = []
    param_top1_list, param_top5_list = [], []
    traj_top1_list, traj_top5_list = [], []
    lr_list = []
    metricL_list = []
    loss_oracle = []
    img_folder = 'ResultsFolder/' + img_name
    if args.gpu == 0:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
        with open('{}/model_args.txt'.format(img_folder), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    model_name = None

    args.img_folder_eval = 'ResultsFolderStatsNew/' + img_name  + '/'

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.is_master):
        if not os.path.exists(args.img_folder_eval):
            os.system('mkdir -p {}'.format(args.img_folder_eval))


    for train_epoch in tqdm(range(train_epoch_, total_epochs)):

        t0 = time.time()
        print('train_epoch', train_epoch)
        if args.distributed:
            train_sampler.set_epoch(train_epoch)
            # test_sampler.set_epoch(train_epoch)
        if (train_epoch % 200 == 100 and train_epoch > 100) or train_epoch == total_epochs - 1 or train_epoch == total_epochs - 20 or train_epoch == total_epochs - 10:
            if args.gpu == 0:
                saved_pth = 'saved_checkpoint_path/{}metric_epch_{}'.format(img_name, train_epoch)
                torch.save({'epoch': train_epoch, 'state_dict':metric_model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, saved_pth)
                saved_pth = 'saved_checkpoint_path/{}param_epch_{}'.format(img_name, train_epoch)
                torch.save({'epoch': train_epoch, 'state_dict':param_model.state_dict(), 'optimizer_state_dict':optimizer.state_dict()}, saved_pth)

                print('emulator model:', saved_pth)
                print('param model:', saved_pth)
                print('lr:', optimizer.param_groups[0]['lr'])
        img_name_ = img_folder + '/epch_{:03d}'.format(train_epoch)
        filter = train_epoch % 50  == 0 and train_epoch > 900

        lr_ = adjust_learning_rate_cos(lr_ori, optimizer, train_epoch, total_epochs, args)
        if alter_tau_metricL:
            epochs = int(args.extra_prefix.split('_')[-1])
            T_metricL_traj_alone = adjust_tau_metricL(T_metricL_ori, train_epoch, total_epochs, epochs = epochs, max_tau_metricL = max_tau_metricL)
            T_metricL_param_alone = T_metricL_traj_alone

            metricL_list.append(T_metricL_traj_alone)
            if train_epoch % 20 == 0:
                plt.plot(np.arange(len(metricL_list)), np.array(metricL_list))
                plt.savefig('{}/metricL.png'.format(img_folder))
                plt.close()
        if alter_tau_inter:
            T_metricL = adjust_tau_metricL(T_metricL_ori, train_epoch, total_epochs, epochs = epochs, max_tau_metricL = max_tau_metricL)
        lr_list.append(lr_)
        metric_model.train()
        param_model.train()

        # pdb.set_trace()
        if train_epoch > train_epoch_:
            topk_traj = traj_head.topk(5, dim = -1)[1].cpu()
            topk_param = param_head.topk(5, dim = -1)[1].cpu()
            param_top1_list.append((topk_param == torch.arange(topk_param.shape[0])[:,None])[:,0].float().mean().item())
            param_top5_list.append(((topk_param == torch.arange(topk_param.shape[0])[:,None]).sum(dim = -1) > 0).float().mean().item())
            traj_top1_list.append((topk_traj == torch.arange(topk_traj.shape[0])[:,None])[:,0].float().mean().item())
            traj_top5_list.append(((topk_traj == torch.arange(topk_traj.shape[0])[:,None]).sum(dim = -1) > 0).float().mean().item())
            if train_epoch % 20 == 0:
                plt.plot(np.arange(len(param_top1_list)), np.array(param_top1_list), label = 'param top1')
                plt.plot(np.arange(len(param_top5_list)), np.array(param_top5_list), label = 'param top5')
                plt.plot(np.arange(len(traj_top1_list)), np.array(traj_top1_list), label = 'traj top1')
                plt.plot(np.arange(len(traj_top5_list)), np.array(traj_top5_list), label = 'traj top5')
                plt.grid(True)
                plt.legend()
                plt.savefig('{}/top_list.png'.format(img_folder))
                plt.close()

        t1 = time.time()
        for i, (anchor_param, anchor_t_crop, pos_param, pos_anchor_t_crop, idx, filter) in enumerate(train_loader):
            print('filter ratio', filter.sum()/anchor_param.shape[0])
            if i == 0:
                idx_list.append(idx.float().mean())

            anchor_param = anchor_param.cuda(args.gpu, non_blocking = True).float()
            anchor_t_crop = anchor_t_crop.cuda(args.gpu, non_blocking = True).float()
            pos_param = pos_param.cuda(args.gpu, non_blocking = True).float()
            pos_anchor_t_crop = pos_anchor_t_crop.cuda(args.gpu, non_blocking = True).float()

            SEP_forward = 'sep' in args.extra_prefix.split('_')
            if SEP_forward:
                mask_zeros = torch.as_tensor(torch.zeros(anchor_t_crop.shape), dtype = torch.int32).to(args.gpu)
                pri_hat_anchor_param_traj, anchor_embed = metric_model.forward(anchor_t_crop[:, None, :, :], mask = mask_zeros[:, None, :, :], train = True, return_head_only = False)
                pri_hat_pos_param_traj, pos_anchor_embed = metric_model.forward(pos_anchor_t_crop[:, None, :, :], mask = mask_ones[:, None, :, :], train = True, return_head_only = False)
            else:
                mask_zeros = torch.as_tensor(torch.zeros(args.batch_size_metricL * 2, args.crop_T, anchor_t_crop.shape[-1]), dtype = torch.int32)
                cat_traj, cat_embed = metric_model.forward(torch.cat([anchor_t_crop[:, None, :, :], pos_anchor_t_crop[:, None, :, :]]), mask = mask_zeros[:, None, :, :], train = True, return_head_only = False)
                pri_hat_anchor_param_traj, pri_hat_pos_param_traj = cat_traj[:args.batch_size_metricL, :], cat_traj[args.batch_size_metricL:, :]
                anchor_embed, pos_anchor_embed = cat_embed[:args.batch_size_metricL, :], cat_embed[args.batch_size_metricL:, :]

            if not SEP_forward:
                cat_param_embed = param_model(torch.cat([anchor_param, pos_param]))
                anchor_param_embed, pos_anchor_param_embed = cat_param_embed[:args.batch_size_metricL, :], cat_param_embed[args.batch_size_metricL:, :]
            else:
                anchor_param_embed = param_model(anchor_param, return_head_only = False)
                pos_anchor_param_embed = param_model(pos_param, return_head_only = False)

            traj_embed_queue = metric_model.module.traj_queue_embed.detach().clone()
            pos_anchor_embed_ = torch.cat([pos_anchor_embed, traj_embed_queue.T], dim = 0)
            param_embed_queue = metric_model.module.param_queue_embed.detach().clone()
            pos_anchor_param_embed_ = torch.cat([pos_anchor_param_embed, param_embed_queue.T], dim = 0)

            sim1 = anchor_embed @ anchor_param_embed.T /  T_metricL
            traj_head = torch.cat([sim1, anchor_embed @ param_embed_queue / T_metricL], dim = -1)
            param_head = torch.cat([sim1.T, anchor_param_embed @ traj_embed_queue / T_metricL], dim = -1)

            labels = torch.arange(traj_head.shape[0], device = args.gpu)
            loss_traj = torch.nn.CrossEntropyLoss()(traj_head, labels) ##fix traj
            loss_param = torch.nn.CrossEntropyLoss()(param_head, labels)

            labels = torch.arange(anchor_embed.shape[0], device = args.gpu)
            sim = (anchor_param_embed @ pos_anchor_param_embed_.T)
            loss_param_alone = torch.nn.CrossEntropyLoss()(sim / T_metricL_param_alone, labels)
            sim = anchor_embed @ pos_anchor_embed_.T
            loss_traj_alone = torch.nn.CrossEntropyLoss()(sim / T_metricL_traj_alone, labels)
            sum_loss = args.loss_fix_param * (loss_traj + loss_param) + \
                   args.loss_param_traj_alone * (args.loss_param_alone_coeff * loss_param_alone + args.loss_traj_alone_coeff * loss_traj_alone)

            pri_hat_param_loss_traj = calculate_parameter_loss(anchor_param, pri_hat_anchor_param_traj, dist_index = args.dist_index, weight_list = [1, 5, 1, 1])

            sum_loss = sum_loss + args.mape_traj_pri * pri_hat_param_loss_traj

            if args.gpu == 0 and train_epoch % 10 == 0:
                print('contrastive embed loss', loss_param_alone, loss_traj_alone)
                if 'partial' in args.extra_prefix.split('_'):
                    print('masked traj alone loss', loss_masked_traj_alone)
                print('fix loss', loss_traj + loss_param)
                # print('param loss', pri_hat_param_loss_traj)
                print('sum loss', sum_loss)

            metric_model.zero_grad()
            param_model.zero_grad()
            sum_loss.backward()
            optimizer.step()

            loss_list_param.append(loss_param.item())
            loss_list_traj.append(loss_traj.item())
            pri_hat_param_mape.append(pri_hat_param_loss_traj.item())
            traj_alone_loss_list.append(loss_traj_alone.item())
            if use_mem_bank:
                with torch.no_grad():
                    if 'partial' not in args.extra_prefix.split('_'):
                        metric_model.module._dequeue_and_enqueue(anchor_param, anchor_param_embed, anchor_embed)
                    else:
                        metric_model.module._dequeue_and_enqueue_masked(anchor_param, anchor_param_embed, anchor_embed, masked_anchor_embed)
            plt.plot(np.arange(len(lr_list)), np.array(lr_list))
            plt.savefig('{}/lr.png'.format(img_folder))
            plt.close()


        if args.mape_traj_pri > 0 and (train_epoch % 200 == 100 or train_epoch == total_epochs - 21 or train_epoch == total_epochs - 11) and \
            train_epoch > 100:
            if args.gpu == 0:
                metric_model.eval()
                param_model.eval()
                run_eval([metric_model, param_model, moment_func, batch_moment_func], args, train_epoch)
                metric_model.train()
                param_model.train()
            torch.distributed.barrier()

        if train_epoch % 20 == 0:
            fig, ax = plt.subplots()
            fig.subplots_adjust(right=0.75)
            twin1 = ax.twinx()
            p1 = ax.plot(np.arange(len(loss_list_param)), np.array(loss_list_param), label = 'param_CE', color = 'blue')
            p2 = twin1.plot(np.arange(len(loss_list_traj)), np.array(loss_list_traj), label = 'traj_CE', color = 'red')
            ax.set_xlabel("epochs", color = 'blue')
            ax.set_ylabel("param_CE", color = 'blue')
            twin1.set_ylabel("traj_CE", color = 'red')
            tkw = dict(size=4, width=1.5)
            ax.tick_params(axis='y', **tkw, labelcolor = 'blue')
            twin1.tick_params(axis='y', **tkw, labelcolor = 'red')
            ax.tick_params(axis='x', **tkw)
            plt.legend()
            plt.savefig('{}/loss.png'.format(img_folder))
            plt.close()

            plt.plot(np.arange(len(idx_list)), np.array(idx_list))
            plt.savefig('{}/idx_list.png'.format(img_folder))
            plt.close()

            plt.plot(np.arange(np.array(pri_hat_param_mape).shape[0]), np.array(pri_hat_param_mape), label = 'traj')
            plt.legend()
            plt.grid(True)
            plt.savefig('{}/pri_hat_mape.png'.format(img_folder))
            plt.close()
            if len(pri_hat_param_mape) > 100:
                plt.plot(np.arange(100), np.array(pri_hat_param_mape)[-100:], label = 'traj')
                plt.legend()
                plt.grid(True)
                plt.savefig('{}/pri_hat_mape_recent.png'.format(img_folder))
                plt.close()
            plt.plot(np.arange(len(param_alone_loss_list)), np.array(param_alone_loss_list), label = 'param_alone')
            plt.plot(np.arange(len(traj_alone_loss_list)), np.array(traj_alone_loss_list), label = 'traj_alone')
            plt.legend()
            plt.savefig('{}/loss_alone'.format(img_folder))
            plt.close()

    metric_model.eval()
    param_model.eval()
    print('finished!!!!!!!!!!')


if __name__ == '__main__':
    args.extra_prefix = args.extra_prefix
    main(args)
