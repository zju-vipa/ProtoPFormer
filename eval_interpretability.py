import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm
from torchvision.datasets.folder import default_loader

import os
import argparse
import protopformer
from tools.preprocess import mean, std
from tools.local_parts import id_to_path, id_to_part_loc, id_to_bbox, in_bbox
from tools.utils import str2bool


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                    bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)


all_colors = [(83, 172, 252), (212, 183, 156), (48, 89, 182), (78, 223, 244), (182, 114, 1),
                (72, 57, 55), (151, 149, 148), (204, 225, 240), (138, 181, 224), (82, 138, 155),
                (169, 219, 161), (126, 137, 235), (112, 160, 0), (166, 106, 146), (108, 57, 209)]


def draw_point(img, point, bbox_size=10, color=(0, 0, 255)):
    img[point[1] - bbox_size // 2: point[1] + bbox_size // 2, point[0] - bbox_size // 2: point[0] + bbox_size // 2] = color
    return img


class Cub2011(Dataset):
    base_folder = 'images'

    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        img_id = sample.img_id

        if self.transform is not None:
            img = self.transform(img)

        return img, target, img_id

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--data_path', type=str)
parser.add_argument('--imgclass', type=int, default=[15,], nargs=1)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--check_test', type=str2bool, default=False)

# Model
parser.add_argument('--data_set', default='CUB2011U', type=str, help='Image Net dataset path')
parser.add_argument('--base_architecture', type=str, default='vgg16')
parser.add_argument('--input_size', default=224, type=int, help='images input size')
parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
parser.add_argument('--prototype_activation_function', type=str, default='log')
parser.add_argument('--add_on_layers_type', type=str, default='regular')

parser.add_argument('--reserve_layers', nargs='+', type=int, default=[])
parser.add_argument('--reserve_token_nums', nargs='+', type=int, default=[])
parser.add_argument('--use_global', type=str2bool, default=False)
parser.add_argument('--use_ppc_loss', type=str2bool, default=False)
parser.add_argument('--ppc_cov_thresh', type=float, default=1.)
parser.add_argument('--ppc_mean_thresh', type=float, default=2.)
parser.add_argument('--global_coe', type=float, default=0.5)
parser.add_argument('--global_proto_per_class', type=int, default=5)

parser.add_argument('--resume', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# specify the test image to be analyzed
test_image_label = args.imgclass[0]
img_size = args.input_size
args.nb_classes = 200
device = torch.device('cuda')

# load the model
args.vis_image = False
check_test_accu = args.check_test
base_architecture = args.base_architecture

mid_dir = 'vanilla' if args.use_ppc_loss is False else 'ppc'
save_analysis_path = os.path.join(args.out_dir, mid_dir, base_architecture)

ppnet = protopformer.construct_PPNet(base_architecture=args.base_architecture,
                                pretrained=True, img_size=args.input_size,
                                prototype_shape=args.prototype_shape,
                                num_classes=args.nb_classes,
                                reserve_layers=args.reserve_layers,
                                reserve_token_nums=args.reserve_token_nums,
                                use_global=args.use_global,
                                use_ppc_loss=args.use_ppc_loss,
                                ppc_cov_thresh=args.ppc_cov_thresh,
                                ppc_mean_thresh=args.ppc_mean_thresh,
                                global_coe=args.global_coe,
                                global_proto_per_class=args.global_proto_per_class,
                                prototype_activation_function=args.prototype_activation_function,
                                add_on_layers_type=args.add_on_layers_type)
if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
ppnet.load_state_dict(checkpoint['model'])

# ppnet = torch.load(args.resume)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

ppnet.eval()

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize
])

test_dataset = Cub2011(args.data_path, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=False, shuffle=False)

num_classes = 200
part_num = 15
half_size = 36
part_thresh = 0.8
num_prototypes = ppnet.num_prototypes
num_prototypes_per_class = ppnet.num_prototypes_per_class
token_reserve_num = ppnet.reserve_token_nums[0]

# Infer on the Test Dataset
all_token_attn, all_proto_acts, all_targets, all_img_ids = [], [], [], []
for idx, (data, targets, img_ids) in tqdm(enumerate(test_loader)):
    
    data = data.cuda()
    targets = targets.cuda()
    token_attn, proto_acts = ppnet.push_forward(data)
    # select the prototypes belonging to its category
    fea_size = proto_acts.shape[-1]
    proto_per_class = 10
    proto_indices = (targets * proto_per_class).unsqueeze(dim=-1).repeat(1, proto_per_class)
    proto_indices += torch.arange(proto_per_class).cuda()
    proto_indices = proto_indices[:, :, None, None].repeat(1, 1, fea_size, fea_size)
    proto_acts = torch.gather(proto_acts, 1, proto_indices)

    all_token_attn.append(token_attn.cpu().detach())
    all_proto_acts.append(proto_acts.cpu().detach())
    all_targets.append(targets.cpu())
    all_img_ids.append(img_ids)
all_token_attn = torch.cat(all_token_attn, dim=0)
all_proto_acts = torch.cat(all_proto_acts, dim=0).numpy()
all_targets = torch.cat(all_targets, dim=0).numpy()
all_img_ids = torch.cat(all_img_ids, dim=0).numpy()

if args.reserve_token_nums[0] != 196:
    sample_num = all_token_attn.shape[0]
    ori_fea_size = int(all_token_attn.shape[-1] ** (1/2))
    all_proto_acts = torch.from_numpy(all_proto_acts)
    token_attn = all_token_attn.reshape(all_token_attn.shape[0], -1)
    # 10 * 10 -> 14 * 14 fill into
    reserve_token_indices = torch.topk(token_attn, k=token_reserve_num, dim=-1)[1]
    reserve_token_indices = reserve_token_indices.sort(dim=-1)[0]
    reserve_token_indices = reserve_token_indices[:, None, :].repeat(1, num_prototypes_per_class, 1)  # (B, 10, 100)
    replace_proto_acts = torch.zeros(sample_num, num_prototypes_per_class, int(ori_fea_size * ori_fea_size))
    replace_proto_acts.scatter_(2, reserve_token_indices, all_proto_acts.flatten(start_dim=2))    # (B, 10, 196)
    replace_proto_acts = replace_proto_acts.reshape(sample_num, num_prototypes_per_class, ori_fea_size, ori_fea_size).numpy()   # (B, 10, 14, 14)
    all_proto_acts = replace_proto_acts

class_proto_effect, class_mean_part, class_max_part = [], [], []
# for test_image_label in args.imgclass:
for test_image_label in tqdm(range(num_classes)):
    arr_ids = np.nonzero(all_targets == test_image_label)[0]
    cur_proto_acts = all_proto_acts[arr_ids, :]
    img_ids = all_img_ids[arr_ids]
    all_proto_to_part, all_part_mask = [], []

    for index, img_id in enumerate(img_ids):
        test_image_path = os.path.join(args.data_path, 'images', id_to_path[img_id][0], id_to_path[img_id][1])
        original_img = cv2.imread(test_image_path)
        img_height, img_width = original_img.shape[0], original_img.shape[1]
        original_img = cv2.resize(original_img, (img_size, img_size))
        prototype_activation_patterns = cur_proto_acts[index : index + 1]

        # Get part labels
        part_labels, part_mask = [], np.zeros(15,)
        bbox = id_to_bbox[img_id]
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        part_locs = id_to_part_loc[img_id]
        for part_loc in part_locs:
            part_id = part_loc[0] - 1   # Begin From 0
            part_mask[part_id] = 1
            ratio_x, ratio_y = part_loc[1] / img_width, part_loc[2] / img_height
            re_loc_x, re_loc_y = int(img_size * ratio_x), int(img_size * ratio_y)
            part_labels.append([part_id, re_loc_x, re_loc_y]) 

        proto_per_class = 10
        proto_to_part = np.zeros((proto_per_class, 15))

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        if args.vis_image:
            img_dir = os.path.join(save_analysis_path, 'correct_class_prototypes', 'category_{}'.format(test_image_label), 'img_{}'.format(index))
            os.makedirs(img_dir, exist_ok=True)
            plt.imsave(os.path.join(img_dir, '0_img_original.jpg'), original_img)

        for prototype_index in range(proto_per_class):
            activation_pattern = prototype_activation_patterns[0, prototype_index]
            activation_value = activation_pattern.max()

            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                    interpolation=cv2.INTER_CUBIC)

            # Show the most highly activated patch of the image by this prototype
            max_indice = np.where(upsampled_activation_pattern==upsampled_activation_pattern.max())
            max_indice = (max_indice[0][0], max_indice[1][0])
            high_act_patch_indices = (max(0, max_indice[0] - half_size), min(img_size, max_indice[0] + half_size), max(0, max_indice[1] - half_size), min(img_size, max_indice[1] + half_size))

            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                        high_act_patch_indices[2]:high_act_patch_indices[3], :]
            
            # Get the related parts of prototype j
            for part_label in part_labels:
                part_id, loc_x, loc_y = part_label[0], part_label[1], part_label[2]
                if in_bbox((loc_y, loc_x), high_act_patch_indices):
                    proto_to_part[prototype_index, part_id] = 1

            if args.vis_image:
                part_img = np.copy(original_img)
                part_img = np.float32(part_img) / 255
                imsave_with_bbox(fname=os.path.join(img_dir, '{}_prototype_bbox.jpg'.format(prototype_index)),
                                    img_rgb=part_img,
                                    bbox_height_start=high_act_patch_indices[0],
                                    bbox_height_end=high_act_patch_indices[1],
                                    bbox_width_start=high_act_patch_indices[2],
                                    bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

                # Save Activation Map
                rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
                rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_activation_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[..., ::-1]
                overlayed_img = 0.7 * part_img + 0.3 * heatmap

                plt.imsave(os.path.join(img_dir, '%d_prototype_acti_%.2f.jpg' % (prototype_index, activation_value)),
                    overlayed_img)

        all_proto_to_part.append(proto_to_part)
        all_part_mask.append(part_mask)
    
    # Calculate Accuracy
    all_proto_to_part = np.stack(all_proto_to_part, axis=0)
    all_proto_to_part = np.transpose(all_proto_to_part, (1, 0, 2))
    all_part_mask = np.stack(all_part_mask, axis=0)

    all_mean_part, all_proto_effect, all_max_part = [], [], []
    for proto_idx in range(all_proto_to_part.shape[0]):
        img_to_part = all_proto_to_part[proto_idx]
        assert ((1. - all_part_mask) * img_to_part).sum() == 0  # Assert that the prototype does not correspond to an object part that cannot be visualized (not in the part annotations)
        img_to_part_sum = img_to_part.sum(axis=0)
        all_part_mask_sum = all_part_mask.sum(axis=0)
        all_part_mask_sum = np.where(all_part_mask_sum == 0, all_part_mask_sum + 1, all_part_mask_sum)  # eliminate the 0 elements in all_part_mask_sum
        mean_part_float = img_to_part_sum / all_part_mask_sum
        mean_part = (mean_part_float >= part_thresh).astype(np.int32)
        all_mean_part.append(mean_part)
        max_part = mean_part_float.max()
        all_max_part.append(max_part)

        if mean_part.sum() == 0:
            all_proto_effect.append(0)
        else:
            all_proto_effect.append(1)
    class_mean_part.extend(all_mean_part)
    class_proto_effect.extend(all_proto_effect)
    class_max_part.extend(all_max_part)
class_mean_part = np.array(class_mean_part)
class_max_part = np.array(class_max_part)
class_proto_effect = np.array(class_proto_effect)
class_proto_effect_score = class_proto_effect.mean()
print('Consistency Score: {:.2%} '.format(class_proto_effect_score))