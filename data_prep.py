import torch
from torch import Tensor
from PIL import Image
from lightning.pytorch import seed_everything
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset

import random
import os
from typing import List, Tuple
from tqdm.auto import tqdm
from glob import glob
from copy import deepcopy
import numpy as np
from numpy.random import default_rng
import pandas as pd





# This code is taken from https://github.com/aliasgharkhani/Masktune/tree/master and modified to fit our needs.

# Some utility functions for data preparation
def filter_data_by_label(data, targets, class_labels_to_filter):
    """
    extract indices of data that have labels that exist in the desired_class_labels
    """
    filtered_target_idx = torch.cat(
        [torch.where(targets == label)[0] for label in class_labels_to_filter]
    )
    return data[filtered_target_idx], targets[filtered_target_idx]


def group_labels(targets, old_to_new_label_mapping):
    """
    assign new labels to data based on the label_grouping
    """
    new_labels = list(old_to_new_label_mapping.keys())
    old_label_groupings = list(old_to_new_label_mapping.values())

    for i, target in enumerate(targets):
        for idx, old_label_grouping in enumerate(old_label_groupings):
            if target in old_label_grouping:
                target = new_labels[idx]

        targets[i] = torch.tensor(int(target))
    return targets


def add_color_bias_to_images(
    class_number: int,
    data: Tensor,
    targets: Tensor,
    bias_conflicting_data_ratio: float,
    bias_colors: List[list]=None,
    bias_type: str="background",
    seed = 0,
    **kwargs,
) -> Tuple[Tensor, List[list]]:
    colors = []
    if class_number == 2:
        bias_colors = [[255, 0, 0], [255, 0, 0]]
    seed_everything(seed)
    color_tensor = torch.randint(0, 256, (class_number, 3), dtype=torch.uint8)
    print(color_tensor)
    for i in range(class_number):
        data_number_to_add_bias = round(len(data[torch.where(targets==i)[0]])*(1-bias_conflicting_data_ratio))
        if class_number == 2 and i == 1:
            data_number_to_add_bias = round(len(data[torch.where(targets==i)[0]])*bias_conflicting_data_ratio)
        target_i_data = data[torch.where(targets==i)[0][:data_number_to_add_bias]]
        if bias_colors is None:
            # color = torch.randint(0, 256, (3,), dtype=torch.uint8)
            color = color_tensor[i, :]
            print("Adding Bias", color)
            colors.append(color.numpy())
        else:
            color = torch.tensor(bias_colors[i], dtype=torch.uint8)
            colors.append(bias_colors[i])
        for j in range(3):
            if bias_type == "background":
                target_i_data[:, :, :, j] = torch.where(target_i_data[:, :, :, j]==0, color[j], target_i_data[:, :, :, j])
            elif bias_type == "foreground":
                target_i_data[:, :, :, j] = torch.where(target_i_data[:, :, :, j]>0, color[j], target_i_data[:, :, :, j])
            elif bias_type == "square":
                target_i_data[:, :kwargs["square_size"], :kwargs["square_size"], j] = torch.ones((len(target_i_data), kwargs["square_size"], kwargs["square_size"])) * color[j]
                if kwargs["square_number"] == 2:
                    target_i_data[:, :kwargs["square_size"], -kwargs["square_size"]:, j] = torch.ones((len(target_i_data), kwargs["square_size"], kwargs["square_size"])) * color[2-j]
        data[torch.where(targets==i)[0][:data_number_to_add_bias]] = target_i_data
    return data, np.array(colors)


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, lengths):
        self.n_holes = n_holes
        self.lengths = lengths

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        c, h, w = img.size()
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            length = random.choice(self.lengths)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class BiasedMNIST(MNIST):
    def __init__(
        self,
        class_labels_to_filter=[i for i in range(0, 10)],
        new_to_old_label_mapping={k:[k] for k in list(range(10))},
        bias_conflicting_data_ratio=0.1,
        bias_type="background",
        square_size=4,
        bias_colors=None,
        square_number=4,
        seed = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train = kwargs["train"]
        self.return_masked = False
        if kwargs["train"]:
            self.split = "train"
        else:
            self.split = "test"
        self.img_data_dir = os.path.join(
            kwargs["root"],
            "BiasedMNIST",
            "images",
            self.split,
            bias_type,
        )
        if bias_type == "square":
            self.img_data_dir = os.path.join(self.img_data_dir, str(square_number))
        self.img_data_dir = os.path.join(self.img_data_dir, f"{round((1-bias_conflicting_data_ratio)*100)}")
        bias_colors_file_path = os.path.join(self.img_data_dir, "bias_colors.npy")
        if not (os.path.isdir(self.img_data_dir) and len(os.listdir(self.img_data_dir)) > 0):
            print(
                f"\n\nstart creating and saving {self.split} dataset of BiasedMnist\n\n"
            )
            os.makedirs(self.img_data_dir, exist_ok=True)
            self.data, self.targets = filter_data_by_label(
                self.data, self.targets, class_labels_to_filter
            )
            self.targets = group_labels(self.targets, new_to_old_label_mapping)
            self.data = torch.unsqueeze(self.data, dim=-1).repeat((1, 1, 1, 3))
            # permute_indices = torch.randperm(len(self.data))
            # self.data = self.data[permute_indices]
            # self.targets = self.targets[permute_indices]
            if bias_type != "none":
                self.data, self.bias_colors = add_color_bias_to_images(
                    len(new_to_old_label_mapping),
                    self.data.clone(),
                    self.targets,
                    bias_conflicting_data_ratio,
                    bias_colors=bias_colors,
                    bias_type=bias_type,
                    square_size=square_size,
                    square_number=square_number,
                    seed = seed,
                )
                np.save(bias_colors_file_path, self.bias_colors)
            for target in list(new_to_old_label_mapping.keys()):
                os.makedirs(os.path.join(self.img_data_dir, str(target)), exist_ok=True)
            for id, (data, target) in enumerate(zip(self.data, self.targets)):
                Image.fromarray(data.numpy().astype(np.uint8)).save(
                    os.path.join(self.img_data_dir, str(target.item()), f"{id}.png")
                )
            self.data = []
            self.targets = []
            print(
                f"\n\nfinished creating and saving {self.split} dataset of BiasedMnist\n\n"
            )
        elif bias_conflicting_data_ratio < 1.0:
            if bias_colors is None:
                self.bias_colors = np.load(bias_colors_file_path)
            else:
                self.bias_colors = bias_colors
                np.save(bias_colors_file_path, self.bias_colors)

        self.update_data(self.img_data_dir)

    def update_data(self, data_file_directory, masked_data_file_directory=None):
        self.data_path = []
        self.data = []
        self.targets = []
        self.masked_data_path = []

        data_classes = sorted(os.listdir(data_file_directory))
        print("-" * 10, f"indexing {self.split} data", "-" * 10)
        for data_class in tqdm(data_classes):
            try:
                target = int(data_class)
            except:
                continue
            class_image_file_paths = glob(
                os.path.join(data_file_directory, data_class, "*")
            )
            for image_file_path in class_image_file_paths:
                temp = Image.open(image_file_path)
                keep = temp.copy()
                self.data.append(keep)
                temp.close()
            self.data_path += class_image_file_paths
            if masked_data_file_directory is not None:
                self.return_masked = True
                masked_class_image_file_paths = sorted(glob(
                    os.path.join(masked_data_file_directory, data_class, '*')))
                self.masked_data_path += masked_class_image_file_paths
            self.targets += [target] * len(class_image_file_paths)


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, img_file_path, target) where target is index of the target class.
        """
        img, img_file_path, target = self.data[index], self.data_path[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_masked:
            masked_img_file_path = self.masked_data_path[index]
            masked_img = Image.open(masked_img_file_path)
            if self.transform is not None:
                masked_img = self.transform(masked_img)
            return img, img_file_path, target, masked_img
        return img, img_file_path, target


# Main class which prepares the data
class MnistTrain():
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def prepare_data_loaders(self, val_indices_path = None) -> None:
        self.transform_test = transforms.Compose([transforms.ToTensor()])
        self.transform_train = transforms.Compose(
            [transforms.ToTensor(), ])

        self.train_dataset = BiasedMNIST(
            root=os.path.join(self.args["base_dir"], "datasets"),
            train=True,
            transform=self.transform_train,
            download=True,
            class_labels_to_filter=[i for i in range(0, 10)],
            new_to_old_label_mapping={
                k:[k] for k in list(range(10))},
            bias_conflicting_data_ratio=self.args["train_bias_conflicting_data_ratio"],
            bias_type=self.args["bias_type"],
            square_number=self.args["square_number"],
            seed = self.args["seed"],
        )

        self.val_dataset = deepcopy(self.train_dataset)
        val_data_dir = self.train_dataset.img_data_dir.replace("train", "val")
        if not (os.path.isdir(val_data_dir) and len(os.listdir(val_data_dir)) > 0):
            print("Path 1")
            os.makedirs(val_data_dir, exist_ok=True)
            for target in list(range(10)):
                os.makedirs(os.path.join(val_data_dir, str(target)), exist_ok=True)
            if val_indices_path is not None:
                val_indices = torch.load(val_indices_path)
                print("Loaded the indices!")
            else:
                rng = default_rng(seed = self.args["seed"])
                val_indices = rng.choice(len(self.train_dataset), size=12000, replace=False)
                print("Generated the indices randomly!", val_indices)
            for val_index in val_indices:
                file_path = self.train_dataset.data_path[val_index]
                target = self.train_dataset.targets[val_index]
                new_file_path = os.path.join(
                    val_data_dir, str(target), file_path.split("/")[-1])
                os.replace(file_path, new_file_path)
            self.train_dataset.update_data(self.train_dataset.img_data_dir)
            self.val_dataset.update_data(val_data_dir)
        else:
            print("Path2")
            self.val_dataset.update_data(val_data_dir)
        if self.args["use_random_masking"]:
            transform_data_to_mask = transforms.Compose(
            [transforms.ToTensor(), Cutout(1, [i for i in range(2, 15)])])
        else:
            transform_data_to_mask = self.transform_train
        self.data_to_mask_dataset = BiasedMNIST(
            root=os.path.join(self.args["base_dir"], "datasets"),
            train=True,
            transform=transform_data_to_mask,
            download=True,
            class_labels_to_filter=[i for i in range(0, 10)],
            new_to_old_label_mapping={
                k:[k] for k in list(range(10))},
            bias_conflicting_data_ratio=self.args["train_bias_conflicting_data_ratio"],
            bias_type=self.args["bias_type"],
            square_number=self.args["square_number"],
            seed = self.args["seed"]+1,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args["train_batch"],
            shuffle=True,
            num_workers=self.args["workers"],
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args["test_batch"],
            shuffle=False,
            num_workers=self.args["workers"],
        )
        self.data_to_mask_loader = torch.utils.data.DataLoader(
            self.data_to_mask_dataset,
            batch_size=self.args["masking_batch_size"],
            shuffle=True,
            num_workers=self.args["workers"],
        )
        self.test_dataset = BiasedMNIST(
            root=os.path.join(self.args["base_dir"], "datasets"),
            train=False,
            transform=self.transform_test,
            download=True,
            class_labels_to_filter=[i for i in range(0, 10)],
            new_to_old_label_mapping={
                k:[k] for k in list(range(10))},
            bias_conflicting_data_ratio=self.args["test_bias_conflicting_data_ratio"],
            bias_type=self.args["test_data_types"],
            square_number=self.args["square_number"],
            seed=self.args["seed"]+2,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args["test_batch"],
            shuffle=False,
            num_workers=self.args["workers"],
        )
        print("-" * 10, "datasets and dataloaders are ready.", "-" * 10)
        


class Rebalanced_Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Get the image path and label at the given index
        image_path, label = self.dataframe.iloc[index]

        # Load the image
        image = Image.open(image_path)

        # Apply transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        return image, image_path, label

