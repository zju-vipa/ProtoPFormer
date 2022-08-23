# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
from statistics import mean
from cv2 import transform
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import scipy.misc
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_url
import matplotlib.pyplot as plt
import scipy.io as sio
from torchvision.datasets import VisionDataset
# import torchvision.datasets as datasets
# from torchvision.datasets.folder import default_loader
# from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive, list_dir
from scipy.io import loadmat
from tools.preprocess import mean as mean_cub
from tools.preprocess import std as std_cub
# from tools.stanford_dogs import Dogs
from typing import Callable, Optional, Any, Tuple
from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
import pathlib


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

def build_dataset_view(is_train, args):
    large_size = int((256 / 224) * args.input_size)
    if 'adam' in args.model[0]:
        transform = transforms.Compose([
            transforms.Resize(size=(args.input_size, args.input_size)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=large_size, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
        ])

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'CUB2011U':
        dataset = Cub2011(args.data_path, train=is_train, transform=transform)
        nb_classes = 200
    elif args.data_set == 'Dogs':
        dataset = Dogs(root=os.path.join(args.data_path, 'stanford_dogs'), train=is_train, cropped=False,
                    transform=transform, download=False)
        nb_classes = 120
    elif args.data_set == 'CUB2011':
        if is_train:
            dataset = datasets.ImageFolder(
                args.train_dir,
                transform=transform)
        else:
            dataset = datasets.ImageFolder(
                args.test_dir,
                transform=transform)
        nb_classes = 200
    elif args.data_set == 'Car':
        split = 'train' if is_train else 'test'
        dataset = StanfordCars(args.data_path, split=split, transform=transform, download=True)
        nb_classes = 196

    return dataset, nb_classes


def build_dataset_noaug(is_train, args):
    large_size = int((256 / 224) * args.input_size)
    transform = []
    # transform.append(transforms.ToTensor())
    if 'adam' in args.model[0]:
        transform.append(transforms.Resize(size=(args.input_size, args.input_size)))
    else:
        transform.append(transforms.Resize(large_size, interpolation=3))
        transform.append(transforms.CenterCrop(args.input_size))
    
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(transform)

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'CUB2011U':
        dataset = Cub2011(args.data_path, train=is_train, transform=transform)
        nb_classes = 200
    elif args.data_set == 'Dogs':
        dataset = Dogs(root=os.path.join(args.data_path, 'stanford_dogs'), train=is_train, cropped=False,
                    transform=transform, download=False)
        nb_classes = 120
    elif args.data_set == 'CUB2011':
        if is_train:
            dataset = datasets.ImageFolder(
                args.train_dir,
                transform=transform)
        else:
            dataset = datasets.ImageFolder(
                args.test_dir,
                transform=transform)
        nb_classes = 200
    elif args.data_set == 'Car':
        split = 'train' if is_train else 'test'
        dataset = StanfordCars(args.data_path, split=split, transform=transform, download=True)
        nb_classes = 196

    return dataset, nb_classes


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == 'FashionMNIST':
        dataset = datasets.FashionMNIST(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'MNIST':
        dataset = datasets.MNIST(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CUB2011U':
        dataset = Cub2011(args.data_path, train=is_train, transform=transform)
        nb_classes = 200
    elif args.data_set == 'Car':
        split = 'train' if is_train else 'test'
        dataset = StanfordCars(args.data_path, split=split, transform=transform, download=True)
        nb_classes = 196
    # elif args.data_set == 'Car':
    #     dataset = Cars(os.path.join(args.data_path, 'stanford-car'), train=is_train,
    #                    transform=transform, download=True)
    #     nb_classes = 196
    elif args.data_set == 'Dogs':
        dataset = Dogs(root=os.path.join(args.data_path, 'stanford_dogs'), train=is_train, cropped=False,
                    transform=transform, download=False)
        nb_classes = 120
    elif args.data_set == 'Caltech256':
        dataset = datasets.Caltech256(args.data_path, transform=transform)
        nb_classes = 256
    elif args.data_set == 'CUB2011':
        normalize = transforms.Normalize(mean=mean_cub,
                                    std=std_cub)
        if is_train:
            dataset = datasets.ImageFolder(
                args.train_dir,
                transforms.Compose([
                    transforms.Resize(size=(args.img_size, args.img_size)),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            dataset = datasets.ImageFolder(
                args.test_dir,
                transforms.Compose([
                    transforms.Resize(size=(args.img_size, args.img_size)),
                    transforms.ToTensor(),
                    normalize,
                ]))
        nb_classes = 200
    elif args.data_set == 'Flowers':
        if is_train:
            data_root = os.path.join(args.data_path, 'Oxford_flower/flowers/train')
        else:
            data_root = os.path.join(args.data_path, 'Oxford_flower/flowers/test')
        dataset = datasets.ImageFolder(root=data_root, transform=transform)
        nb_classes = 102
    elif args.data_set == 'FGVC':
        dataset = Aircraft(root=args.data_path, train=is_train,
                       transform=transform, download=False)
        nb_classes = 100
    elif args.data_set == 'WikiArt':
        root = os.path.join(args.data_path, 'wikiart', 'train' if is_train else 'test')
        dataset = ImageFolder(root=root, transform=transform)
        nb_classes = 195
    elif args.data_set == 'Sketches':
        root = os.path.join(args.data_path, 'sketches', 'train' if is_train else 'test')
        dataset = ImageFolder(root=root, transform=transform)
        nb_classes = 250
    elif args.data_set == 'Places365':
        root = os.path.join(args.data_path, 'places365_standard', 'train' if is_train else 'val')
        dataset = ImageFolder(root=root, transform=transform)
        nb_classes = 365


    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET12':
        if os.path.isdir(os.path.join(args.data_path, 'ILSVRC2012', 'train')):
        # root = os.path.join(args.data_path, 'ILSVRC2012', 'train_set' if is_train else 'val_set')
            root = os.path.join(args.data_path, 'ILSVRC2012', 'train' if is_train else 'val')
        elif os.path.isdir(os.path.join(args.data_path, 'ILSVRC2012', 'train_set')):
            root = os.path.join(args.data_path, 'ILSVRC2012', 'train_set' if is_train else 'val_set')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'CIFAR32':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100

    elif args.data_set == 'CIFAR100_CNN':
        dataset = get_cifar_100_224(
            root=args.data_path,
            split=is_train,
        )
        nb_classes = 100

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        if args.data_set in ['FashionMNIST', 'MNIST']:
            # channel == 1
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=(0.485,),
                std=(0.229,),
            )
        else:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
        if not resize_im and args.input_size == 32:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        # elif not resize_im and args.input_size == 28:
        #     # replace RandomResizedCropAndInterpolation with
        #     # RandomCrop
        #     transform.transforms[0] = transforms.RandomCrop(
        #         args.input_size, padding=4)
                
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if args.data_set in ['FashionMNIST', 'MNIST']:
        t.append(transforms.Normalize((0.485,), (0.229,)))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def get_cifar_100_224(root: str, split: str = "train") -> Dataset:
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])
    if split == "train":
        transform = train_transform
        is_train = True
    else:
        transform = test_transform
        is_train = False

    target_transform = None
    dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    return dataset


def get_cifar_100(root: str, split: str = "train") -> Dataset:
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if split == "train":
        transform = train_transform
        is_train = True
    else:
        transform = test_transform
        is_train = False

    target_transform = None
    dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    return dataset


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
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

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target


    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()


class Cars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    file_list = {
        'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims.tgz'),
        'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
    }

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(Cars, self).__init__(root,)

        self.loader = default_loader
        self.train = train
        self.transform = transform

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        loaded_mat = sio.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
        loaded_mat = loaded_mat['annotations'][0]
        self.samples = []
        for item in loaded_mat:
            if self.train != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.samples.append((path, label))

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                and os.path.exists(os.path.join(self.root, self.file_list['annos'][1])))

    def _download(self):
        print('Downloading...')
        for url, filename in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
        print('Extracting...')
        archive = os.path.join(self.root, self.file_list['imgs'][1])
        extract_archive(archive)


class Dogs(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'dog'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self, root, train=True, cropped=False, transform=None, target_transform=None, download=False):
        # self.root = join(os.path.expanduser(root), self.folder)
        self.root = root
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = os.path.join(self.root, 'Images')
        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(os.path.join(self.annotations_folder, annotation))]
                                        for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [(annotation+'.jpg', idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', idx) for annotation, idx in split]

            self._flat_breed_images = self._breed_images

        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(os.path.join(self.root, 'Images')) and os.path.exists(os.path.join(self.root, 'Annotation')):
            if len(os.listdir(os.path.join(self.root, 'Images'))) == len(os.listdir(os.path.join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + os.path.join(self.root, tar_filename))
            with tarfile.open(os.path.join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(os.path.join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0]-1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self._flat_breed_images), len(counts.keys()), float(len(self._flat_breed_images))/float(len(counts.keys()))))

        return counts


class Aircraft(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None,
                 target_transform=None, download=False):
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.loader = default_loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s...' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print('Extracting %s...' % tar_path)
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images



def move_images(image_dir, image_labels):
    """
    Download at http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
        image_dir: 'Dataset images'
        image_labels: 'The image labels'
    Moves images:
        From: image_dir
        To: flowers/train, flowers/valid and flowers/test directory (ratio: 5/7, 1/7, 1/7).
            Subdirectories labeled with category (1-102).
            Each category is different species of flower.
    """

    if image_dir[-1] != '/':
        image_dir = image_dir + '/'

    # Make training, validation and testing directories
    os.mkdir('/nfs/xmq/data/dataset/flowers102/flowers')
    train_dir = '/nfs/xmq/data/dataset/flowers102/flowers/train/'
    test_dir = '/nfs/xmq/data/dataset/flowers102/flowers/test/'
    valid_dir = '/nfs/xmq/data/dataset/flowers102/flowers/valid/'
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(valid_dir)

    # Make subdirectories
    for label in range(1, 103):
        os.mkdir(train_dir + '/' + str(label))
        os.mkdir(test_dir + '/' + str(label))
        os.mkdir(valid_dir + '/' + str(label))

    # Load labels (image index to category)
    labels = loadmat(image_labels, squeeze_me=True)
    labels_np = np.array(labels['labels'])

    # Move images to new subdirectories
    last_l = 0
    for i, l in enumerate(labels_np):
        if last_l != l:
            count = 1
            last_l = l
        else:
            count += 1

        label = str(l)
        i_str = str(i + 1)
        if len(i_str) == 1:
            src_start = image_dir + 'image_0000'
            dst_start = '/image_0000'
        elif len(i_str) == 2:
            src_start = image_dir + 'image_000'
            dst_start = '/image_000'
        elif len(i_str) == 3:
            src_start = image_dir + 'image_00'
            dst_start = '/image_00'
        else:
            src_start = image_dir + 'image_0'
            dst_start = '/image_0'

        src = src_start + i_str + '.jpg'
        if count % 6 == 0:
            dst = valid_dir + label + dst_start + i_str + '.jpg'
        elif count % 7 == 0:
            dst = test_dir + label + dst_start + i_str + '.jpg'
        else:
            dst = train_dir + label + dst_start + i_str + '.jpg'

        os.rename(src, dst)
    os.rmdir(image_dir)



if __name__ == '__main__':
    '''
    dataset = CUB(root='./CUB_200_2011')

    for data in dataset:
        print(data[0].size(),data[1])

    '''

    # import argparse
    # import helper

    # parser = argparse.ArgumentParser(
    #     description='Move image downloads to newly created flowers directory,'
    #                 ' delete download directory.')
    # parser.add_argument('image_dir', action="store", default='/nfs/xmq/data/dataset/flowers102',
    #                     help='image directory path')
    # parser.add_argument('labels_file', action="store",
    #                     default='/nfs/xmq/data/dataset/flowers102/imagelabels.mat',
    #                     help='labels file path')
    # args = parser.parse_args()

    print('Moving files..')
    image_dir = '/nfs/xmq/data/dataset/flowers102/jpg'
    labels_file = '/nfs/xmq/data/dataset/flowers102/imagelabels.mat'
    move_images(image_dir, labels_file)
    print('flowers directory created')

    exit()


    # data_root = '/nfs/xmq/data/dataset/stanford_cars'
    data_root = '/datasets/fine-grain_dataset/StanfordDogs'
    trainset = Dogs(root=data_root,
                            train=True,
                            cropped=False,
                            download=False
                            )
    testset = Dogs(root=data_root,
                            train=False,
                            cropped=False,
                            download=False
                            )
    print(len(trainset), len(testset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0,
                                              drop_last=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=True, num_workers=0,
                                              drop_last=True)
    print(len(trainloader), len(testloader))
    exit()


    train_dataset = Cars('/nfs/xmq/data/dataset/stanford-car', train=True, download=False)
    test_dataset = Cars('/nfs/xmq/data/dataset/stanford-car', train=False, download=False)
    print(len(train_dataset), len(test_dataset))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0,
                                              drop_last=True)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0,
                                              drop_last=True)

    dataset_train, nb_classes = build_dataset(is_train=True)
    dataset_val, _ = build_dataset(is_train=False)


    # Read the data set in the way of DataLoader in pytorch
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    datapath = '/nfs/xmq/data/dataset'
    train_dataset = Cub2011(root=datapath, train=True, transform=transform_train,)
    print(len(train_dataset))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0,
                                              drop_last=True)
    print(len(trainloader))

    test_dataset = Cub2011(root=datapath, train=False, transform=transform_train,)
    print(len(test_dataset))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0,
                                              drop_last=False)    
    
    print(len(testloader))
