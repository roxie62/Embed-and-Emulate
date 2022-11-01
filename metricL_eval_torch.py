import torch
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from tqdm import tqdm
from numpy.random import multivariate_normal as mv_normal
import time
from configuration import *


@torch.no_grad()
def xi(z, H_perp, x, model, args, use_ode_solver = False):
    theta = z @ H_perp.T if z.shape[1] != 4 else z
    try:
        batch_param = theta.clone()
    except:
        theta = torch.from_numpy(theta)
        batch_param = theta.clone().float()
    batch_param[:, 2] = torch.exp(batch_param[:, 2])

    with torch.no_grad():
        model.eval()
        f_mean_hat = model(batch_param)
    mask = f_mean_hat.sum(dim = -1).isnan()
    f_mean_hat = f_mean_hat[~mask]
    theta = theta[~mask]

    out = torch.cat([theta, f_mean_hat], dim = -1)
    return out

def run_eki(metric_model, model, img_name, x, args, gt_idx, gt_param = np.array([10, 1, 10, 10]), \
            E = 100, N = 40, use_regression_head = True, save_result = False):
    metric_model.eval()
    Z_pri, f = metric_model(torch.from_numpy(x)[None, :, :].float().to(args.gpu), mask = mask, return_head_only = False)
    f_mean = f.mean(dim = 0)
    f_mean = torch.nn.functional.normalize(f_mean, dim = -1)[:, None]

    n_params = gt_param.shape[0]
    n_moments = len(f_mean)
    n_vars = n_params + n_moments

    H = torch.from_numpy(np.eye(n_moments, n_vars, n_params)).to(args.gpu).float()
    H_perp = torch.from_numpy(np.eye(n_params, n_vars)).to(args.gpu).float()
    I = torch.eye(n_vars).to(args.gpu).float()

    r = 0.3
    R = r**2 * torch.eye(f_mean.shape[0]).to(args.gpu).float()

    Z = np.random.normal(0, 1, (n_params, E))
    ratio = 1
    Z *= np.sqrt([ratio*36, ratio*2.25, ratio*0.15, ratio*36]).reshape(-1, 1)
    c_mean = np.log(11.5)
    Z += np.array([7.5, 2.5, c_mean, 12.5])[:, None]

    Z = torch.from_numpy(Z).to(args.gpu).float()
    Z_exp = Z.cpu().data.numpy()
    Z_exp[2, :] = np.exp(Z_exp[2, :])
    Z_std = Z_exp.std(axis = 1)

    results_theta = []
    results_mape = []

    # change prior distribution
    # Z_pri = metric_model(torch.from_numpy(x)[None, :, :].float().to(args.gpu), return_params_only = True)
    Z_pri = Z_pri.squeeze().cpu().data.numpy()
    mape = abs(Z_pri.mean(axis = 0) - gt_param.cpu().data.numpy()) / (abs(gt_param.cpu().data.numpy()) + np.array([0.1, 0.1, 0.1, 0.1]))
    Z_mean = Z_pri.mean(axis = 0)
    Z_mean_ = Z_mean.copy()

    if use_regression_head:
        Z_mean[2] = np.log(Z_mean[2])
        Z = torch.from_numpy(np.random.normal(0, 1, (n_params, E))).to(args.gpu).float()
        # Z *= torch.from_numpy(np.sqrt([20, 4, 0.15, 20]).reshape(-1, 1)).to(args.gpu).float()
        ratio = 0.5
        Z *= torch.from_numpy(np.sqrt([36*ratio, 2.25*ratio, 0.15*ratio, 36*ratio]).reshape(-1, 1)).to(args.gpu).float()
        Z += torch.from_numpy(Z_mean.reshape(-1, 1)).to(args.gpu).float()
        Z_exp = Z.cpu().data.numpy()
        Z_exp[2, :] = np.exp(Z_exp[2, :])
        Z_std = Z_exp.std(axis = 1)
        print(Z_std)

    if use_enki:
    for _ in range(N):
        USE_thershold = True
        if USE_thershold:
            gt_param_min = torch.from_numpy(np.array([-5, 0, 0.1, 0])[:, None].repeat(E, 1)).float().to(args.gpu)
            gt_param_min[2, :] = torch.log(gt_param_min[2, :])
            gt_param_max = torch.from_numpy(np.array([20, 5, 25, 25])[:, None].repeat(E, 1)).float().to(args.gpu)
            gt_param_max[2, :] = torch.log(gt_param_max[2, :])
            if _ == 0:
                Z[Z < gt_param_min] = gt_param_min[Z < gt_param_min]
                Z[Z > gt_param_max] = gt_param_max[Z > gt_param_max]
            else:
                Theta = H_perp @ Z
                Theta[Theta < gt_param_min] = gt_param_min[Theta < gt_param_min]
                Theta[Theta > gt_param_max] = gt_param_max[Theta > gt_param_max]
                Z[:4, :] = Theta
        Z = xi(Z.T, H_perp, x, model).T
        z_bar = Z.mean(dim=1)
        C = torch.mean(torch.stack([torch.outer(z, z) for z in Z.T]), dim=0)
        C -= torch.outer(z_bar, z_bar)
        K = C @ H.T @ torch.inverse(H @ C @ H.T + R)
        E = Z.shape[1]
        O = f_mean + torch.from_numpy(mv_normal(np.zeros(f_mean.shape[0]), R.cpu().data.numpy(), E).T).to(args.gpu).float()
        Z = (I - K @ H) @ Z + K @ O
        Theta = H_perp @ Z
        Theta[2, :] = torch.exp(Theta[2, :])
        theta_mean = Theta.mean(dim=1)
        theta_std = Theta.std(dim = 1)
        theta_iqr = Theta.std(dim = 1)
        results_theta.append(Theta)
        enki_mape = 100 * abs(theta_mean.cpu() - gt_param)/ (abs(gt_param) + torch.tensor([0.1, 0.1, 0.1, 0.1]))
        regression_mean = Z_mean_
        enki_last_round = results_theta[-1].T
    if save_result:
        print(enki_mape)
        torch.save(Z_pri, 'regression_head_results_noise_{}'.format(args.add_noise_alpha))
        torch.save(torch.stack(results_theta).cpu().data.numpy(), 'enki_results_noise_{}'.format(args.add_noise_alpha))

    return enki_mape, theta_mean, regression_mean, enki_last_round, N, r, Z_std


@torch.no_grad()
def create_eval_eki_with_metric_model(args, save_list = True):
    def run_eki_func(metric_model, model, img_name, args, idx = False):

        total_gt_params_list = []
        averaged_mape = []
        mape_run_list = []
        wrong_list = []
        use_regression_head = True
        E, N = 100, 50
        # E, N = 10000, 100
        if not use_regression_head:
            E, N = 10000, 100
            # E, N = 1000, 100
            E, N = 100, 100

        use_enki = True
        nan_list = []
        mape_list_regression = []
        estimate_param_all_list = []
        gt_param_all_list = []
        regression_list = []
        os.system('rm test_ig_l96/eval*.png')
        import glob

        data_path = '/net/scratch/roxie62/emulator/testing_data_new_v5'
        data_list = glob.glob('{}/long_000*.pth'.format(data_path))
        data_list.sort()
        if 'test' in args.extra_prefix.split('_'):
            # data_list = data_list[:249]
            data_list = data_list[12:261]

        idx_list = data_list
        gt_path = True
        index_permutation = np.load('index_permute.npy')
        load_file = save_list
        extra_prefix = args.extra_prefix

        if args.gpu == 0 and load_file:
            stats_folder = '/net/scratch/roxie62/emulator/stats_folder/'
            train_size = args.train_size
            with open('{}/{}_{}_{}_E{}_N{}{}.txt'.format(stats_folder, train_size, 'E&E', use_regression_head, E, N, extra_prefix), 'w') as f:
                f.writelines('begin evaluation \n')

        test_the_oracle = False
        len_test = 1 if test_the_oracle else len(idx_list)
        count = 0
        sum_diff = 0

        for idx in range(len_test):
            gt_idx = idx
            args.gt_idx = int(idx_list[idx].split('.')[-2].split('_')[-1])
            ipath = idx_list[idx]
            print('gt idx', gt_idx)
            mape_list = []
            if args.add_noise_alpha < 0.1:
                seed_number = 1
            else:
                seed_number = 1
            for seed in range(1):
                if test_the_oracle:
                    oracle = torch.load('the_oracle.pth')
                    gt_param = torch.tensor([10, 1, 10, 10])
                else:
                    oracle_d = torch.load(ipath)
                    assert oracle_d['1'].success == True
                    gt_param, oracle = oracle_d['0'], oracle_d['1'].y.T

                mask = None

                filter_x = oracle[-500:, :36].reshape(-1).std() < 5e-5
                filter_y = oracle[-500:, 36:].reshape(-1).std() < 5e-5

                filter_nan = filter_x and filter_y
                if filter_nan:
                    nan_list.append(gt_idx)
                else:
                    oracle = torch.from_numpy(oracle).cuda(args.gpu).float()
                    K = 36
                    add_noise = test_the_oracle
                    seed = 0
                    torch.random.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    total_gt_params_list.append(gt_param.cpu().data.numpy())
                    if add_noise:
                        noise_alpha = args.add_noise_alpha
                        print('noise alpha is', noise_alpha, args.add_noise_alpha)
                        alpha_y = 1
                        traj = oracle.clone()[None, :, :]
                        mean = traj.mean(dim = 0)
                        std = traj.std(dim = 1)[:, None, :].repeat(1, traj.shape[1], 1)
                        noise_alpha_mask = noise_alpha

                        noise_traj = torch.cat([(noise_alpha_mask * std * (torch.randn(traj.shape, device = traj.device)))[:, :, :K], \
                                alpha_y * (noise_alpha_mask * std * (torch.randn(traj.shape, device = traj.device)))[:, :, K:]], dim = -1)


                        oracle = oracle + noise_traj.squeeze()
                    oracle = oracle.cpu().data.numpy()

                    enki_mape, theta_mean, regression_mean, enki_last_round, N, r, Z_std = run_eki(metric_model, model, img_name, oracle, args, \
                                                gt_idx, mask = mask, gt_param = gt_param,
                                                seed = seed, use_regression_head = use_regression_head, use_enki = use_enki, save_result = test_the_oracle, \
                                                E = E, N = N)
                    print(enki_mape)
                    estimate_param_all_list.append(enki_last_round.cpu().data.numpy())
                    gt_param_all_list.append(gt_param)

                    regression_list.append(regression_mean)
                    if use_enki:
                        mape_regression = 100 * (abs(regression_mean - gt_param.cpu().data.numpy())) / (abs(gt_param) + np.array([0.1, 0.1, 0.1, 0.1]))
                        mape_list_regression.append(mape_regression.cpu().data.numpy())
                    mape_list.append(enki_mape.cpu().data.numpy())

                    if test_the_oracle:
                        print('regression estimates', np.round(regression_mean, 2))
                        print('enki estimates', np.round(theta_mean.cpu().data.numpy(), 2))

                    if load_file:
                        wrong_list.append(gt_idx)
                        sum1 = np.round(mape_regression.sum(),2)
                        sum2 = np.round(enki_mape.sum().cpu().data.numpy(), 2)
                        sum_diff += sum2 - sum1
                        if sum2 > sum1:
                            count += 1

                        with open('{}/{}_{}_{}_E{}_N{}{}.txt'.format(stats_folder, train_size, 'E&E', use_regression_head, E, N, extra_prefix), 'a+') as f:
                            f.writelines('---------------------gt_idx {} \n'.format(gt_idx))
                            f.writelines('gt_param {} \n'.format(gt_param.cpu().data.numpy()))
                            f.writelines('regression estimates {} \n'.format(np.round(regression_mean, 2)))
                            f.writelines('regression mape {} sum {} \n'.format(np.round(mape_regression, 2), np.round(mape_regression.sum(),2)))
                            f.writelines('Z_std {} \n'.format(Z_std))
                            f.writelines('EnKI std {} \n'.format(enki_last_round.std(axis = 0)))
                            f.writelines('enki estimates {} \n'.format(np.round(theta_mean.cpu().data.numpy(), 2)) )
                            f.writelines('enki mape {} sum {} \n'.format(np.round(np.mean(np.array(mape_list).reshape(-1, 4), axis = 0), 2), np.round(enki_mape.sum().cpu().data.numpy(), 2)))
                            f.writelines('enki larger number {}, sum diff {} \n'.format(count, sum_diff))
                            f.writelines('\n')
                    averaged_mape.append(mape_list)

        # print('nan list', nan_list)
        averaged_regression = np.array(mape_list_regression).reshape(-1, 4).mean(axis = 0)
        median_regression = np.median(np.array(mape_list_regression).reshape(-1, 4), axis = 0)
        print('averaged regression:', averaged_regression)
        print('median regression:', median_regression)
        averaged_enki = np.array(averaged_mape).reshape(-1, 4).mean(axis = 0)
        median_enki = np.median(np.array(averaged_mape).reshape(-1, 4), axis = 0)
        print('averaged:', averaged_enki)
        print('median:', median_enki)
        print('mape 25 quantile', np.quantile(np.array(averaged_mape).reshape(-1, 4), 0.25, axis = 0))
        print('mape 75 quantile', np.quantile(np.array(averaged_mape).reshape(-1, 4), 0.75, axis = 0))
        # print('the list of index with large mape sum', wrong_list)
        print(len(averaged_mape))

        if load_file:
            with open('{}/{}_{}_{}_E{}_N{}{}.txt'.format(stats_folder, train_size, 'E&E', use_regression_head, E, N, extra_prefix), 'a+') as f:
                f.writelines('averaged regression {} \n'.format(np.round(averaged_regression, 2)))
                f.writelines('median regression {} \n'.format(np.round(median_regression, 2)))
                f.writelines('averaged enki {} \n'.format(np.round(averaged_enki, 2)))
                f.writelines('median enki {} \n'.format(np.round(median_enki, 2)))
                f.writelines('25 percentile {} \n'.format(np.round(np.quantile(np.array(averaged_mape).reshape(-1, 4), 0.25, axis = 0), 2)))
                f.writelines('75 percentile {} \n'.format(np.round(np.quantile(np.array(averaged_mape).reshape(-1, 4), 0.75, axis = 0),2)))
                f.writelines('the list of index with large mape sum {} \n'.format(wrong_list))
                f.writelines('length of testing params {} '.format(np.array(averaged_mape).reshape(-1, 4).shape[0]))



        print(args.add_noise_alpha)
        if save_list and use_regression_head:
            torch.save({'regression_list':np.array(regression_list).reshape(-1, 4), \
                        'estimate_param_all_list': estimate_param_all_list, 'total_gt_params_list':total_gt_params_list, \
                        'averaged regression': averaged_regression, 'median regression':median_regression, \
                        'averaged enki':averaged_enki, 'median enki':median_enki}, \
                        'enki_eval_with_contrastive_model/results_trainsize{}_noise{}'.format(args.train_size, args.add_noise_alpha))

        return averaged_enki, median_enki, averaged_regression, median_regression


    return run_eki_func
