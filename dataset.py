import os
import torch
import typing
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from augmentations import load_image
from augmentations import load_mosaic
from augmentations import augment_hsv

# ---- BACKBONE SECTION ----


class PILLfeats(torch.utils.data.Dataset):
    """ArcFace dataset

    extract 40x40 px image and class label of it(8 classes by fraction pill space to 1600)
    """

    def __init__(self, df, transforms=None):
        self.df = df

        self.transforms = transforms

        self._items = list(zip(df['path'], df['class']))
        self.classes_to_samples = {}

        a = self.df.to_dict()['class']
        self.classes_to_samples = {i: [] for i in range(9)}
        for i, j in zip(a.values(), a.keys()):
            i = (i, 7)[i > 7]
            self.classes_to_samples[i].append(j)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        im_path, label = self._items[index]
        label = (label, 7)[label > 7]

        img = np.asarray(Image.open(im_path))
        img = self.transforms(image=img)[
            'image'] if self.transforms is not None else img
        img = torch.from_numpy(img.copy()).permute((2, 0, 1)).float()

        return img, torch.tensor(label, dtype=torch.long)


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """Random dataset sampler

    Creates batch with uniform distribution of classes
    """

    def __init__(self, data_source, elems_per_class):
        self.trainset = data_source
        self.el_per_cl = elems_per_class
        self.cl_per_batch = 8

        sample = []
        self.batch_size = self.el_per_cl*self.cl_per_batch
        length = len(self.trainset)//self.batch_size

        for j in range(length):
            sample.append([])
            classes = np.arange(self.cl_per_batch)
            for i in classes:
                elems = np.random.choice(
                    self.trainset.classes_to_samples[i], self.el_per_cl)
                for v in elems:
                    sample[j].append(v)
        self.sample = sample

    def __len__(self):
        return len(self.sample)

    def __iter__(self):
        for i in self.sample:
            yield i

# ---- DETECTION SECTION ----


class DetectionExtractor(Dataset):
    """Return dataset

    Creates detection's dataset
    - img_dir  : root_path for images(root_path/0.jpg, ...)
    - label_dir: root_path for labels(root_path/0.txt, ...)
                > each row => class_id, x, y, w, h 
    """

    def __init__(self, img_dir: str, label_dir: str, transforms=None, rotate=False):
        # Default image size
        self.img_size = 640

        # Default FEATURE MAP size
        self.feature_size = 19

        # Anchors for detection 136/2000 = 40/640 = 0.063
        self.anchor = (0.063, 0.063)

        # Use for mosaic a quater of original image
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]

        # Img's path list
        self.imgs = []

        # Label's list
        self.labels = []

        self.indices = []

        # Rotation boolean flag
        self.rotate = rotate

        for idx, (img_path, label_path) in enumerate(zip(sorted(os.listdir(img_dir)), sorted(os.listdir(label_dir)))):
            self.indices.append(idx)

            # Full image path
            img = os.path.join(img_dir, img_path)
            self.imgs.append(img)

            # In TRAIN mode add 3 additional copies for rotations(90, 180, 270) and labels
            if self.rotate:
                self.imgs.append(img)
                self.imgs.append(img)
                self.imgs.append(img)

                rot90_labels = []
                rot180_labels = []
                rot270_labels = []

            cur_labels = []

            with open(os.path.join(label_dir, label_path)) as label_txt:
                for row in label_txt.readlines():
                    # Make label fields(except class_id) float
                    x, y, w, h = map(float, row.split()[1:])
                    cur_labels.append([
                        x,
                        y,
                        w,
                        h,
                        1.0,
                        0,
                        0
                    ])

                    if self.rotate:
                        rot90_labels.append([
                            y,
                            1-x,
                            h,
                            w,
                            1.0,
                            0,
                            1
                        ])
                        rot180_labels.append([
                            1-x,
                            1-y,
                            w,
                            h,
                            1.0,
                            0,
                            2
                        ])
                        rot270_labels.append([
                            1-y,
                            x,
                            h,
                            w,
                            1.0,
                            0,
                            3
                        ])

            self.labels.append(np.array(cur_labels))

            if self.rotate:
                self.labels.append(np.array(rot90_labels))
                self.labels.append(np.array(rot180_labels))
                self.labels.append(np.array(rot270_labels))

        self.len = len(self.labels)
        self.transform = transforms

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.imgs[idx]
        labels_out = self.labels[idx]

        if self.transform:
            labels = []

            if random.random() < 0.5:
                # Load mosaic
                img, labels4 = load_mosaic(self, idx)
                W, H, _ = img.shape

                for i in labels4:
                    # Because of incorrect albumentations bbox perform RandomCrop in train_det_ts was changed to CenterCrop
                    # and bbox filter implemented in the section below
                    if i[0] > W // 4 and i[1] > H // 4 and i[2] < W * 3 // 4 and i[3] < H * 3 // 4:
                        new_lst = i.copy()
                        w, h = new_lst[2] - new_lst[0], new_lst[3] - new_lst[1]
                        center = [new_lst[0] + w // 2, new_lst[1] + h // 2]
                        new_lst[:4] = [*center, w, h]
                        new_lst[:4] /= W

                        if new_lst[0] > 0 and new_lst[0] < 1 and new_lst[1] > 0 and new_lst[1] < 1:
                            labels.append(new_lst)

            else:
                img, _, _ = load_image(self.imgs[idx])
                img = np.rot90(img, k=labels_out[0, -1]).copy()
                labels = self.labels[idx].copy()

            labels = np.array(labels)

            augmentation = self.transform(image=np.asarray(img), bboxes=labels)
            img = augmentation['image']
            labels = augmentation['bboxes']

            # Fix labels(out of bounds)
            correct_labels = []
            for row in labels:
                if row[2] >= 0.062*0.51 and row[3] >= 0.062*0.51:
                    correct_labels.append(row)

            labels = np.array(correct_labels)

            # Update after albumentations
            nl = len(labels)

            # HSV color-space
            augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4)

            # Flip left-right
            if random.random() < 0.5:
                img = np.fliplr(img)
                if nl:
                    labels[:, 0] = 1 - labels[:, 0]

            labels_out = torch.zeros((nl, 6))
            if nl:
                labels_out = torch.from_numpy(labels)
        else:
            img, _, _ = load_image(self.imgs[idx])
            if self.rotate:
                img = np.rot90(img, k=labels_out[0, -1])
            labels_out = torch.from_numpy(labels_out.copy())

        img = img.transpose((2, 0, 1))[::-1]
        img = torch.from_numpy(np.asarray(img).copy()).float()
        labels_out = self.create_feature_map(labels_out).float()

        return img, labels_out

    def create_feature_map(self, labels):
        # Extract only [x, y, w, h]
        box_map = labels[:, :4]
        box_map[:, 0] *= self.feature_size
        box_map[:, 1] *= self.feature_size

        # Set masks and target values for each gt
        box_col = box_map[:, 0].clamp(0, self.feature_size-1).long()
        box_row = box_map[:, 1].clamp(0, self.feature_size-1).long()

        pred_mask = torch.zeros(
            (5, self.feature_size, self.feature_size), dtype=labels.dtype)

        # Confidience
        pred_mask[0, box_row, box_col] = labels[:, 4]
        # X center - col
        pred_mask[1, box_row, box_col] = box_map[:, 0] - \
            box_col.float()
        # Y center - row
        pred_mask[2, box_row, box_col] = box_map[:, 1] - \
            box_row.float()
        # Width center - scale
        pred_mask[3, box_row, box_col] = (
            box_map[:, 2] / self.anchor[0]).log()
        # Height center - scale
        pred_mask[4, box_row, box_col] = (
            box_map[:, 3] / self.anchor[1]).log()

        return pred_mask


class DetectionInference(Dataset):
    """Return dataset

    Creates detection's dataset
    - img_path  : root_path for images(root_path/0.jpg, ...)
    - rotate   : boolean flag for image rotation
    """

    def __init__(self, img_path: str, rotate=True):
        # Default image size
        self.img_size = 640

        # Img's path list
        self.imgs = []

        # Label's list
        self.labels = []

        # Rotation boolean flag
        self.rotate = rotate

        # Full image path
        img = img_path
        self.imgs.append(img)

        # In TRAIN mode add 3 additional copies for rotations(90, 180, 270) and labels
        if self.rotate:
            self.imgs.append(img)
            self.imgs.append(img)
            self.imgs.append(img)

            rot90_labels = []
            rot180_labels = []
            rot270_labels = []

        cur_labels = []

        cur_labels.append([0])
        self.labels.append(np.array(cur_labels))

        if self.rotate:
            rot90_labels.append([1])
            rot180_labels.append([2])
            rot270_labels.append([3])

            self.labels.append(np.array(rot90_labels))
            self.labels.append(np.array(rot180_labels))
            self.labels.append(np.array(rot270_labels))

        self.len = len(self.labels)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        import cv2

        img = self.imgs[idx]
        labels_out = self.labels[idx]

        img, _, _ = load_image(self.imgs[idx])
        img = cv2.resize(img, (640, 640))

        if self.rotate:
            # Only one value is rotation degree
            img = np.rot90(img, k=labels_out[0, -1])

        img = img.transpose((2, 0, 1))[::-1]
        img = torch.from_numpy(np.asarray(img).copy()).float()

        return img
