import cv2
import random
import numpy as np
import albumentations as A

from utils import load_image
from utils import xywhn2xyxy


train_arcface_tsfm = A.Compose(
    [
        A.RandomRotate90(),
        A.Flip(),
        #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3), - DEPRECATED
    ]
)

test_arcface_tsfm = A.Compose(
    []
)

train_det_ts = A.Compose(
    [
        A.CenterCrop(640, 640),
        A.Blur(p=0.1),
        A.MedianBlur(p=0.1),
        A.ToGray(p=0.1),
        A.CLAHE(p=0.1),
    ],
    bbox_params=A.BboxParams(format='yolo')
)


def load_mosaic(self, index):
    #  4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4 = []
    s = self.img_size

    yc, xc = (int(random.uniform(-x, 2 * s + x))
              for x in self.mosaic_border)  # mosaic center x, y

    # 3 additional image indices
    indices = [index] + random.choices(self.indices, k=3)
    random.shuffle(indices)

    for i, index in enumerate(indices):
        # Load image
        # By default - rotation degree(0, 90, 180, 270) in label
        img, _, (h, w) = load_image(self.imgs[index])

        # Labels
        labels = self.labels[index].copy()
        img = np.rot90(img, k=labels[0, -1]).copy()

        # place img in img4
        if i == 0:  # top left
            # base image with 4 tiles(filled by medium color - 114)
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            # xmin, ymin, xmax, ymax (large image)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            # xmin, ymin, xmax, ymax (small image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        if labels.size:
            # normalized xywh to pixel xyxy format
            labels = xywhn2xyxy(labels, w, h, padw, padh)
        labels4.append(labels)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)

    return img4, labels4


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * \
            [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(
            sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)
