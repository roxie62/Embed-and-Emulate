import torch, pdb, os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from train_utils import compute_averaged_moment_batch
from mpdb import mpdb
import torch.nn.utils.spectral_norm as spectral_norm
import math

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class skip_embed_final(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, norm_layer, args):
        super().__init__()
        self.embed = nn.Sequential(
                               nn.Linear(input_dim, latent_dim),
                               norm_layer(latent_dim),
                               ACT(inplace = True),
                               nn.Linear(latent_dim, output_dim),
                               norm_layer(output_dim),
                               )
        self.skip = nn.Linear(input_dim, output_dim)
        self.final_layer = nn.Identity()

    def forward(self, input):
        return self.final_layer(self.embed(input) + self.skip(input))


# 512-512-64 finalbn
class skip_embed_final_shallow(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, norm_layer, args):
        super().__init__()
        self.embed = nn.Sequential(
                               nn.Linear(input_dim, output_dim),
                               norm_layer(output_dim),
                               ACT(inplace = True),
                               )
        self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return self.embed(input) + self.skip(input)


class skip_embed_start_kse(torch.nn.Module):
    ## this block is applied only when input_dim is lower than output_dim.
    def __init__(self, relu_dim, norm_layer, args):
        super().__init__()
        self.param_proj = nn.Sequential(*[nn.Linear(1, relu_dim) for i in range(3)])
        self.embed = nn.Sequential(
                               norm_layer(3 * relu_dim),
                               ACT(inplace = True),
                               )
        self.skip_param_proj = nn.Sequential(*[nn.Linear(1, relu_dim) for i in range(3)])

    def forward(self, input):
        embed_feat = self.embed(torch.cat([self.param_proj[i](input[:,i]) for i in range(3)], dim = -1))
        skip_feat = torch.cat([self.skip_param_proj[i](input[:,i]) for i in range(3)], dim = -1)
        out = embed_feat + skip_feat
        out = self.ACT_embed(out)
        return embed_feat + skip_feat

class skip_embed_start(torch.nn.Module):
    ## this block is applied only when input_dim is lower than output_dim.
    def __init__(self, relu_dim, norm_layer, args):
        super().__init__()
        self.param_proj = nn.Sequential(*[nn.Linear(1, relu_dim) for i in range(4)])
        self.embed = nn.Sequential(
                               norm_layer(4 * relu_dim),
                               ACT(inplace = True),
                               )
        self.skip_param_proj = nn.Sequential(*[nn.Linear(1, relu_dim) for i in range(4)])

    def forward(self, input):
        embed_feat = self.embed(torch.cat([self.param_proj[i](input[:,i]) for i in range(4)], dim = -1))
        skip_feat = torch.cat([self.skip_param_proj[i](input[:,i]) for i in range(4)], dim = -1)
        out = embed_feat + skip_feat
        return out

class skip_embed(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, norm_layer, args):
        super().__init__()
        self.embed = nn.Sequential(
                               nn.Linear(input_dim, latent_dim),
                               norm_layer(latent_dim),
                               ACT(inplace = True),
                               nn.Linear(latent_dim, output_dim),
                               norm_layer(output_dim),
                               )

    def forward(self, input):
        out = self.embed(input) + input
        return out

ACT = nn.ReLU

class ParamEmbed(torch.nn.Module):
    def __init__(self, args, external_final_embed_layer = None):
        super().__init__()
        latent_dim = args.embed_dim
        first_proj_dim = args.hidden_dim_param
        relu_dim = first_proj_dim  / 4 if args.l96 else first_proj_dim  / 3
        norm_layer = nn.BatchNorm1d
        activation = nn.ReLU
        self.args = args
        num_of_param = 4 if args.l96 else 3
        latent_width = args.hidden_dim_param
        if 'shallow' in self.args.extra_prefix.split('_'):
            print('using shallow network')
            self.embed = nn.Sequential(
                            skip_embed_start(int(relu_dim), norm_layer, args),
                            skip_embed(latent_width, latent_width, latent_width, norm_layer, args),
                            skip_embed(latent_width, latent_width, latent_width, norm_layer, args),
                            skip_embed(latent_width, latent_width, latent_width, norm_layer, args),
                            skip_embed(latent_width, latent_width, latent_width, norm_layer, args),
                            skip_embed(latent_width, latent_width, latent_width, norm_layer, args),
                            skip_embed_final_shallow(latent_width, latent_width, latent_dim, norm_layer, args),
                            )

    def forward(self, param, return_head_only = True):
        embed = self.embed(param[:, :, None])
        return F.normalize(embed, dim = -1)


class MetricNet(torch.nn.Module):
    """
    Network module for a single level.
    """

    def __init__(self, T, args, use_moment = False, use_bn = True, use_sn = False):
        super().__init__()
        self.T = T
        print('PAY attention, crop size is', T)
        self.args = args
        norm_layer = nn.BatchNorm2d
        num_of_param = 4 if args.l96 else 3
        latent_dim = args.embed_dim
        first_dim = 11
        if args.l96:
            from resnet import resnet34
            self.encoder = resnet34(num_classes = num_of_param, first_dim = first_dim, norm_layer = norm_layer)
        else:
            from kse_resnet import resnet34
            self.encoder = resnet34(num_classes = num_of_param, norm_layer = norm_layer)
        dim_mlp = self.encoder.fc.weight.shape[1]
        activation = nn.ReLU

        if args.use_bn_embed:
            self.proj_head = skip_embed_final(512, 512, latent_dim, nn.BatchNorm1d, args)
        else:
            self.proj_head = nn.Sequential(nn.Linear(512, latent_dim))

        # create the queue
        if args.l96:
            self.register_buffer("param_value_queue", torch.randn(4, args.bank_size))
        else:
            self.register_buffer("param_value_queue", torch.randn(3, args.bank_size))
        self.register_buffer("param_queue_embed", torch.randn(latent_dim, args.bank_size))
        self.register_buffer("traj_queue_embed", torch.randn(latent_dim, args.bank_size))
        self.param_queue_embed = nn.functional.normalize(self.param_queue_embed, dim=0)
        self.traj_queue_embed = nn.functional.normalize(self.traj_queue_embed, dim=0)
        self.masked_traj_queue_embed = nn.functional.normalize(self.traj_queue_embed, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, param_value_keys, param_embed_keys, traj_embed_keys):
        # gather keys before updating queue
        param_value_keys = concat_all_gather(param_value_keys)
        param_embed_keys = concat_all_gather(param_embed_keys)
        traj_embed_keys = concat_all_gather(traj_embed_keys)

        batch_size = traj_embed_keys.shape[0]
        K = self.param_queue_embed.shape[1]
        ptr = int(self.queue_ptr)
        assert K % batch_size == 0  # for simplicity
        self.param_value_queue[:, ptr:ptr + batch_size] = param_value_keys.T
        self.param_queue_embed[:, ptr:ptr + batch_size] = param_embed_keys.T
        self.traj_queue_embed[:, ptr:ptr + batch_size] = traj_embed_keys.T
        ptr = (ptr + batch_size) % K  # move pointer


        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_masked(self, param_value_keys, param_embed_keys, traj_embed_keys, masked_traj_embed_keys):
        # gather keys before updating queue
        param_value_keys = concat_all_gather(param_value_keys)
        param_embed_keys = concat_all_gather(param_embed_keys)
        traj_embed_keys = concat_all_gather(traj_embed_keys)
        masked_traj_embed_keys = concat_all_gather(masked_traj_embed_keys)

        batch_size = traj_embed_keys.shape[0]
        K = self.param_queue_embed.shape[1]
        ptr = int(self.queue_ptr)
        assert K % batch_size == 0  # for simplicity
        self.param_value_queue[:, ptr:ptr + batch_size] = param_value_keys.T
        self.param_queue_embed[:, ptr:ptr + batch_size] = param_embed_keys.T
        self.traj_queue_embed[:, ptr:ptr + batch_size] = traj_embed_keys.T
        self.masked_traj_queue_embed[:, ptr:ptr + batch_size] = masked_traj_embed_keys.T
        ptr = (ptr + batch_size) % K  # move pointer

        self.queue_ptr[0] = ptr


    def crop_for_test(self, traj):
        T = traj.shape[1]
        num_of_crop = int(T / self.T)
        extra_T = T % self.T
        Tsidx = torch.arange(num_of_crop) * self.T + extra_T
        test_crop = []
        for i in range(num_of_crop):
            test_crop.append(traj[:,Tsidx[i]:Tsidx[i]+self.T])
        test_crop = torch.stack(test_crop, dim = 1)
        print('test crop', test_crop.shape)
        return test_crop

    def forward(self, traj, mask = 0, train = False, return_params_only = False, return_head_only = True):
        # c: number of crops
        # s: crop T
        # h: dim of l96 (396)

        if not train:
            crop = self.crop_for_test(traj)
        else:
            crop = traj
        n,c, s,h = crop.shape
        if not self.args.kse:
            crop_x = crop[...,:36].reshape(n * c, s, 36, 1)
            crop_y = crop[...,36:].reshape(n * c, s, 36, 10)
            crop = torch.cat([crop_x, crop_y], dim = -1).permute(0,3,2,1)
        else:
            crop = crop.reshape(n*c,s,1,-1).permute(0,2,3,1) ## crop: torch.Size([64, 11, 36, 200])

        x, params = self.encoder(crop)
        embed = self.proj_head(x)
        if return_head_only and not return_params_only:
            return F.normalize(embed, dim = -1).reshape(n,c,-1)
        if return_params_only:
            return params.reshape(n, c, -1)
        return params, F.normalize(embed, dim = -1)



# utils
@torch.no_grad()
def concat_all_gather(tensor):
     """
     Performs all_gather operation on the provided tensors.
     *** Warning ***: torch.distributed.all_gather has no gradient.
     """
     tensors_gather = [torch.ones_like(tensor)
         for _ in range(torch.distributed.get_world_size())]
     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

     output = torch.cat(tensors_gather, dim=0)
     return output
