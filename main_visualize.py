from lib2to3.pgen2 import token
from turtle import distance
from sklearn.inspection import permutation_importance
import torch
import random
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

import re

import os
import pickle

from matplotlib import cm
import protopformer

from tools.preprocess import preprocess_input_function
from tools.datasets import build_dataset, build_dataset_noaug, build_dataset_view
from tools.utils import str2bool

import argparse


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_discard_img(view_img, discard_indices, fea_size, patch_size, replace_color):
    res_img = np.copy(view_img)
    for discard_indice in discard_indices:
        indice_h, indice_w = discard_indice // fea_size, discard_indice % fea_size
        res_img[indice_h * patch_size: (indice_h + 1) * patch_size, indice_w * patch_size: (indice_w + 1) * patch_size] = replace_color
    return res_img


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def get_gaussian_params(proto_act, scale_coe=0.9):
    fea_size = proto_act.shape[-1]
    discrete_values = np.array([[x, y] for x in range(fea_size) for y in range(fea_size)])
    discrete_values = discrete_values.transpose(1, 0) # (2, 196)
    weights = proto_act.flatten()[np.newaxis, :]   # (1, 196)
    weights = weights / weights.sum(axis=-1)
    weights *= fea_size * fea_size
    # weights = (weights / weights.sum(axis=-1)) * scale_coe  # (1, 196)
    
    value_mean = np.mean(discrete_values * weights, axis=-1)    # (2,)
    cut_value = discrete_values - value_mean[:, np.newaxis]
    value_cov = np.dot(cut_value * weights, cut_value.transpose(1, 0))
    value_cov /= (fea_size * fea_size - 1) # (2, 2)

    return value_mean, value_cov


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


def save_fig(X, Y, Z, save_path):
    # scale Z
    Z = Z * 100

    # plot using subplots
    fea_size = X.shape[0]
    fig = plt.figure()
    ax1 = fig.gca(projection='3d')

    surf = ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=2, antialiased=True,
                    cmap=cm.viridis)
    # ax1.view_init(elev=75, azim=0)
    # ax1.view_init(elev=45, azim=20)
    ax1.view_init(elev=10, azim=20)

    ax1.set_xticks(np.arange(0, 14, 4))
    ax1.set_yticks(np.arange(0, 14, 4))
    # ax1.set_zticks(np.array([4, 8]))
    ax1.spines['bottom'].set_linewidth(8)
    ax1.spines['left'].set_linewidth(8)

    f_size = 20
    ax1.set_xlabel(r'$x^2$', fontsize=f_size, labelpad=12)
    ax1.set_ylabel(r'$x^1$', fontsize=f_size, labelpad=12)
    ax1.set_zlabel(r'similarity score', fontsize=20, labelpad=5)
    # fig.colorbar(surf, location='right', shrink=0.5, aspect=5)
    plt.subplots_adjust(left=0, bottom=0.05, right=1, top=0.95, hspace=0.1, wspace=0.1)
    plt.xticks(fontsize=f_size)
    plt.yticks(fontsize=f_size)
    ax1.tick_params('z', labelsize=20)

    # figure = plt.gcf()
    plt.savefig(save_path, pad_inches=-1)
    plt.clf()


def get_args():
    # Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs=1, type=str, default='0')
    parser.add_argument('--modeldir', nargs=1, type=str)
    parser.add_argument('--model', nargs=1, type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # visual args
    parser.add_argument('--finetune', type=str, default='')
    parser.add_argument('--visual_type', type=str, default='heatmap')
    # dataset
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_set', default='CIFAR100', 
        choices=['CIFAR100', 'CIFAR10', 'CUB2011', 'CUB2011U', 'Dogs', 'Caltech256', 'Car'],
        type=str, help='Image Net dataset path')
    parser.add_argument('--data_path', type=str, default='./datasets/cub200_cropped/')
    # ProptoNet
    parser.add_argument('--base_architecture', type=str, default='vgg16')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 192, 1, 1])
    parser.add_argument('--reserve_layers', nargs='+', type=int, default=[])
    parser.add_argument('--reserve_token_nums', nargs='+', type=int, default=[])
    parser.add_argument('--use_global', type=str2bool, default=False)
    parser.add_argument('--use_ppc_loss', type=str2bool, default=False)
    parser.add_argument('--global_coe', type=float, default=0.3)
    parser.add_argument('--global_proto_per_class', type=int, default=10)
    parser.add_argument('--output_dir', default='output_view/')
    parser.add_argument('--use_gauss', type=str2bool, default=False)
    parser.add_argument('--vis_classes', nargs='+', type=int, default=[6, 7, 8])

    parser.add_argument('--prototype_activation_function', type=str, default='log')
    parser.add_argument('--add_on_layers_type', type=str, default='regular')
    #parser.add_argument('-dataset', nargs=1, type=str, default='cub200')
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.0)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    """
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    """

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    args = parser.parse_args()

    return args

args = get_args()
set_seed(1028)

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
device = torch.device(args.device)

# Dataset
dataset_train, args.nb_classes = build_dataset_noaug(is_train=True, args=args)
dataset_train_view, args.nb_classes = build_dataset_view(is_train=True, args=args)
dataset_val, _ = build_dataset_noaug(is_train=False, args=args)
dataset_val_view, _ = build_dataset_view(is_train=False, args=args)

sampler_train = torch.utils.data.SequentialSampler(dataset_train)
sampler_train_view = torch.utils.data.SequentialSampler(dataset_train_view)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
sampler_val_view = torch.utils.data.SequentialSampler(dataset_val_view)

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False,
    shuffle=False
)

train_view_loader = torch.utils.data.DataLoader(
    dataset_train_view,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False,
    shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False,
    shuffle=False
)

test_view_loader = torch.utils.data.DataLoader(
    dataset_val_view,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False,
    shuffle=False
)

# load the model
load_model_dir = args.modeldir[0]
load_model_name = args.model[0]
load_model_path = os.path.join(load_model_dir, load_model_name)
ppnet = protopformer.construct_PPNet(base_architecture=args.base_architecture,
                                pretrained=True, img_size=args.img_size,
                                prototype_shape=args.prototype_shape,
                                num_classes=args.nb_classes,
                                reserve_layers=args.reserve_layers,
                                reserve_token_nums=args.reserve_token_nums,
                                use_global=args.use_global,
                                use_ppc_loss=args.use_ppc_loss,
                                global_coe=args.global_coe,
                                global_proto_per_class=args.global_proto_per_class,   # delete
                                prototype_activation_function=args.prototype_activation_function,
                                add_on_layers_type=args.add_on_layers_type)

print('load model from ' + load_model_path)
load_model = torch.load(load_model_path, map_location='cuda:0')
if 'model' in load_model.keys():
    ppnet.load_state_dict(load_model['model'], strict=False)
else:
    ppnet.load_state_dict(load_model, strict=False)
ppnet = ppnet.cuda()
ppnet.eval()

patch_size = 16
proto_per_category = 10
use_train_imgs = False
view_loader = test_view_loader if use_train_imgs is False else train_view_loader
loader = test_loader if use_train_imgs is False else train_view_loader

select_colors = np.array([[47, 243, 224], [250, 38, 160], [248, 210, 16], [245, 23, 32]])
token_reserve_num = args.reserve_token_nums[-1]

for category_id in args.vis_classes:

    print('process category {}...'.format(category_id))
    data_dir = os.path.join(args.output_dir, args.data_set)
    addstr = 'train' if use_train_imgs else 'test'
    model_dir = os.path.join(data_dir, args.base_architecture + '-{}'.format(args.finetune) + '-{}'.format(addstr))
    visual_dir = os.path.join(model_dir, args.visual_type)
    category_dir = os.path.join(visual_dir, 'category_{}'.format(category_id))
    os.makedirs(category_dir, exist_ok=True)

    view_imgs, view_labels = [], None
    for i, (x, y) in enumerate(view_loader):
        # img = x[0].permute(1, 2, 0) * 255
        # img = img.numpy().astype(np.int32)
        # cv2.imwrite('output_view2/test.jpg', img)

        imgs = x.permute(0, 2, 3, 1) * 255
        imgs = imgs.numpy().astype(np.uint8)
        for k in range(len(imgs)):
            imgs[k] = cv2.cvtColor(imgs[k], cv2.COLOR_BGR2RGB)

        view_imgs.append(imgs)
        if view_labels is None:
            view_labels = y.numpy()
        else:
            view_labels = np.concatenate([view_labels, y.numpy()], axis=0)
        if len(np.nonzero(view_labels == category_id)[0]) > 20:
            break
    view_imgs = np.concatenate(view_imgs, axis=0)

    labels = view_labels

    # model inference to get activation maps on last layer
    labels = None
    pred_labels = None
    all_token_attn, min_distances, all_cls_attn = [], [], []
    for i, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()
        logits, auxi_items = ppnet.forward(x)
        token_attn, distances = auxi_items[0], auxi_items[1]
        _, pred = logits.topk(k=1, dim=1)

        all_token_attn.append(token_attn.detach().cpu().numpy())
        min_distances.append(distances.detach().cpu().numpy())
        # labels.append(y.numpy())
        if labels is None:
            labels = y.cpu().numpy()
            pred_labels = pred.cpu().numpy()
        else:
            labels = np.concatenate([labels, y.cpu().numpy()], axis=0)
            pred_labels = np.concatenate([pred_labels, pred.cpu().numpy()], axis=0)
        if len(np.nonzero(labels == category_id)[0]) > 20:
            break
    all_token_attn = np.concatenate(all_token_attn, axis=0)
    distances = np.concatenate(min_distances, axis=0)
    proto_acts = np.log((distances + 1) / (distances + ppnet.epsilon))
    total_proto_acts = proto_acts

    proto_acts = np.amax(proto_acts, (2, 3))
    last_layer = ppnet.last_layer.weight.detach().cpu().numpy().transpose(1, 0)  # (2000, 200)
    logits = np.dot(proto_acts, last_layer)

    # get attn
    select_idxes = np.nonzero(labels == category_id)[0]
    cur_token_attn = all_token_attn[select_idxes]
    cur_labels = pred_labels[select_idxes]
    total_proto_acts = total_proto_acts[select_idxes]
    logits = logits[select_idxes]
    is_pred_right = (cur_labels == category_id)
    sample_num, num_prototypes = total_proto_acts.shape[0], total_proto_acts.shape[1]

    # reserve_num = [x[1] for x in reserve_layer_nums]
    fea_size = int(cur_token_attn.shape[-1] ** (1/2))
    token_attn = cur_token_attn.reshape(cur_token_attn.shape[0], -1)
    token_attn = torch.from_numpy(token_attn)

    # 9 * 9 -> 14 * 14 fill into
    total_proto_acts = torch.from_numpy(total_proto_acts)
    reserve_token_indices = torch.topk(token_attn, k=token_reserve_num, dim=-1)[1]
    reserve_token_indices = reserve_token_indices.sort(dim=-1)[0]
    reserve_token_indices = reserve_token_indices[:, None, :].repeat(1, num_prototypes, 1)  # (B, 2000, 81)
    replace_proto_acts = torch.zeros(sample_num, num_prototypes, 196)
    replace_proto_acts.scatter_(2, reserve_token_indices, total_proto_acts.flatten(start_dim=2))    # (B, 2000, 196)
    replace_proto_acts = replace_proto_acts.reshape(sample_num, num_prototypes, 14, 14).numpy()

    view_imgs = view_imgs[select_idxes]
    for proto_idx in range(10):
    # for proto_idx in range(1):
        # select view_imgs, proto_acts, labels
        print('process proto {}...'.format(proto_idx))
        cur_proto_acts = replace_proto_acts[:, category_id * proto_per_category + proto_idx]
        heatmaps, patch_idxes, proto_acts, all_coors = [], [], [], []
        for k in range(len(cur_proto_acts)):
            proto_act = cur_proto_acts[k]

            new_proto_act = proto_act
            new_proto_act = new_proto_act - np.amin(new_proto_act)
            new_proto_act = new_proto_act / np.amax(new_proto_act)
            new_proto_act = cv2.applyColorMap(np.uint8(255 * new_proto_act), cv2.COLORMAP_JET)
            proto_acts.append(new_proto_act)

            # save gaussian activation map
            if args.use_gauss:
                gaussian_proto_act = np.copy(proto_act)
                fea_size = gaussian_proto_act.shape[-1]
                act_mean, act_cov = get_gaussian_params(gaussian_proto_act)  # (2,), (2, 2)
                X = np.linspace(0, fea_size - 1, 150)
                Y = np.linspace(0, fea_size - 1, 150)
                X, Y = np.meshgrid(X, Y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = multivariate_gaussian(pos, act_mean, act_cov)
                img_dir = os.path.join(category_dir, 'img_{}'.format(k))
                if os.path.exists(img_dir) is False:
                    os.makedirs(img_dir, exist_ok=True)
                save_fig(X, Y, Z, os.path.join(img_dir, 'gaussian_{}.jpg'.format(proto_idx)))

            upsampled_act = cv2.resize(proto_act, (args.input_size, args.input_size), interpolation=cv2.INTER_CUBIC)
            upsampled_act = upsampled_act - np.amin(upsampled_act)
            upsampled_act = upsampled_act / np.amax(upsampled_act)

            # get the top 5% bounding box
            coor = find_high_activation_crop(upsampled_act)

            heatmap = cv2.applyColorMap(np.uint8(255 * upsampled_act), cv2.COLORMAP_JET)
            patch_idx = [t[0] for t in np.where(proto_act == proto_act.max())]
            heatmaps.append(heatmap)
            patch_idxes.append(patch_idx)
            all_coors.append(coor)
        heatmaps = np.stack(heatmaps, axis=0)
        proto_acts = np.stack(proto_acts, axis=0)
        acti_imgs = (view_imgs * 0.7 + heatmaps * 0.3).astype(np.uint8)
        
        # view the masks
        if args.visual_type == 'slim_gaussian':
            # draw the bounding boxes
            for k in range(len(view_imgs)):
                img_dir = os.path.join(category_dir, 'img_{}'.format(k))
                if os.path.exists(img_dir) is False:
                    os.makedirs(img_dir, exist_ok=True)
                bnd_img = view_imgs[k]
                coor = all_coors[k]
                start_point, end_point = (coor[2], coor[0]), (coor[3], coor[1]) # (x coor, y coor)

                # part_img = bnd_img[coor[0]:coor[1], coor[2]:coor[3]]
                # cv2.imwrite(os.path.join(img_dir, 'proto{}_reserve{}_part.jpg'.format(proto_idx, token_reserve_num)), part_img)
                bnd_img = cv2.rectangle(bnd_img.astype(np.uint8).copy(), start_point, end_point, (0, 255, 255), thickness=2)
                cv2.imwrite(os.path.join(img_dir, 'proto{}_reserve{}_bnd.jpg'.format(proto_idx, token_reserve_num)), bnd_img)

            num_patches = token_attn.shape[-1]
            replace_color = [0, 0, 0]
            discard_token_indices = torch.topk(token_attn, k=num_patches - token_reserve_num, dim=-1, largest=False)[1]
            final_imgs, discard_imgs = [], []
            for k in range(len(acti_imgs)):
                cur_discard_indices = discard_token_indices[k].numpy()

                discard_img = get_discard_img(view_imgs[k], cur_discard_indices, fea_size, patch_size, replace_color)
                acti_img = acti_imgs[k]
                final_imgs.append(acti_img)
                discard_imgs.append(discard_img)

            for k in range(len(final_imgs)):
                img_dir = os.path.join(category_dir, 'img_{}'.format(k))
                if os.path.exists(img_dir) is False:
                    os.makedirs(img_dir, exist_ok=True)
                cv2.imwrite(os.path.join(img_dir, 'proto{}_reserve{}.jpg'.format(proto_idx, token_reserve_num)), final_imgs[k])
                
                discard_path = os.path.join(img_dir, 'catch_img_reserve{}_mask.jpg'.format(token_reserve_num))
                if os.path.exists(discard_path) is False:
                    cv2.imwrite(discard_path, discard_imgs[k])