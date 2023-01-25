import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import albumentations as A
from matplotlib import pyplot as plt


def load_image(path):
    # loads 1 image, returns im, original hw, resized hw
    # BGR
    im = cv2.imread(path)
    assert im is not None, f'Image Not Found {path}'

    # Original shape
    h0, w0 = im.shape[:2]

    # RETURNS: im, hw_original, hw_resized
    return im, (h0, w0), im.shape[:2]


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h, ...] normalized to [x1, y1, x2, y2, ...] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)

    # top left x
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw
    # top left y
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh
    # bottom right x
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw
    # bottom right y
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh

    return y


def rot_90(feature_map, k: int = 0):
    """Rotate feature map (5x19x19) k times

    feature_map - result of YoloEmb [conf, x, y, w, h]
    k - num of rotations
    """

    result = np.zeros(feature_map.shape, dtype=feature_map.dtype)

    # Rotate feature map
    rotated_fmap = np.rot90(feature_map, k=k, axes=(1, 2)).copy()
    result[0] = rotated_fmap[0]

    # Correct values (feature_map[1] = x_center -> new_x_center, ...)
    if k == 1:
        new_x = rotated_fmap[2]
        new_y = 1-rotated_fmap[1]
        new_w = rotated_fmap[4]
        new_h = rotated_fmap[3]
    elif k == 2:
        new_x = 1-rotated_fmap[1]
        new_y = 1-rotated_fmap[2]
        new_w = rotated_fmap[3]
        new_h = rotated_fmap[4]
    elif k == 3:
        new_x = 1-rotated_fmap[2]
        new_y = rotated_fmap[1]
        new_w = rotated_fmap[4]
        new_h = rotated_fmap[3]
    else:
        new_x = rotated_fmap[1]
        new_y = rotated_fmap[2]
        new_w = rotated_fmap[3]
        new_h = rotated_fmap[4]

    result[1] = new_x
    result[2] = new_y
    result[3] = new_w
    result[4] = new_h

    return result


def nms(pred, threshold=0.5, visual: bool = False, original_size=640):
    """Non-Maximum Suppression
    - pred      - YoloEmb featrue map
    - threshold - if confidenct > threshold then use this prediction
    - visual    - plot result feature map
    """
    result_feature_map = np.zeros((original_size, original_size))

    # In this solution only one anchor is needed
    anchor = (0.063, 0.063)

    # Additional map for correct detections coordinates
    range_x, range_y = np.meshgrid(
        np.arange(19, dtype=pred.dtype),
        np.arange(19, dtype=pred.dtype),
    )

    X_shifted = range_x + pred[1]
    Y_shifted = range_y + pred[2]

    # Use idxs where confidence > threshold
    thresholded_idxs = pred[0] > threshold

    labels = pred[0][thresholded_idxs]

    xs = X_shifted[thresholded_idxs]
    ys = Y_shifted[thresholded_idxs]

    # Original coordinates: original_image_size * shifted_idxs / feature_map_size
    xs = (original_size * xs / 19).astype(np.int32)
    ys = (original_size * ys / 19).astype(np.int32)

    # Original detections width and height:
    # - width  = original_image_size * anchor_width * exp(predicted_width_multiplier)
    # - height = original_image_size * anchor_height * exp(predicted_height_multiplier)
    ws = (original_size * anchor[0] * np.exp(pred[3]
          [thresholded_idxs])).astype(np.uint8)
    hs = (original_size * anchor[1] * np.exp(pred[4]
          [thresholded_idxs])).astype(np.uint8)

    zipped_gen = zip(labels, xs, ys, ws, hs)

    nms_list = []
    for conf, x, y, w, h in sorted(zipped_gen, key=lambda l: l[0], reverse=True):
        # In this solution iou between different objects must be zero
        if result_feature_map[y - h // 2:y + h // 2, x - w // 2:x + w // 2].sum() == 0:
            result_feature_map[y - h // 2:y +
                               h // 2, x - w // 2:x + w // 2] = 1
            nms_list.append([conf, x, y, w, h])

    # Convert preditcion to uint type for good plots
    result_feature_map = result_feature_map.astype(np.uint8)

    if visual:
        plt.imshow(result_feature_map)
        plt.show()

    return result_feature_map, nms_list


def create_arc_dataset(filename="marked.csv", savedir="arc", train=True):
    """Creates ArcFace dataset

    filename - path to csv file with descriptions
    savedir - prefix of saving directory and result csv: savedirTrain / savedirVal
    """

    arc_dataset_creation_transform = A.Compose(
        [
            A.Resize(640, 640),
        ]
    )

    df = pd.read_csv(filename)
    data = {'path': [], 'class': []}

    root = f'./{savedir}'
    save_arc_csv_name = f'{savedir}'

    if train:
        root += "Train/"
        save_arc_csv_name += "Train.csv"
    else:
        root += "Val/"
        save_arc_csv_name += "Val.csv"

    if not os.path.exists(root):
        os.makedirs(root)

    idx_generator = range(df.shape[0]-150)
    if not train:
        idx_generator = range(df.shape[0]-150, df.shape[0]-100)

    for i in tqdm(idx_generator):
        im = np.asarray(Image.open(df['img'][i]))
        seg = np.asarray(Image.open(df['seg'][i]))

        transformed = arc_dataset_creation_transform(image=im, mask=seg)

        im = transformed['image']
        seg = transformed['mask']

        for j in range(31):
            for k in range(31):
                idx = i*31*31+31*j+k
                path = os.path.join(root, str(idx))+'.jpg'

                cur_im = im[j*20:j*20+40, k*20:k*20+40]
                cur_seg = seg[j*20:j*20+40, k*20:k*20+40]

                if cur_seg.max() == 255:
                    cur_seg = cur_seg/255

                if len(cur_seg.shape) > 2:
                    cur_seg = cur_seg[..., 0]

                label = int(cur_seg.sum()//158)
                data['class'].append(label)
                data['path'].append(path)

                save_im = Image.fromarray(cur_im)
                save_im.save(path)

    df_arc = pd.DataFrame(data)
    df_arc.to_csv(save_arc_csv_name, index=False)


def convert_to_original(pred, img, threshold=0.5, visual: bool = False, nms_flag: bool = False, save: bool = False, fname: str = None, original_size=640):
    """Plot results with image and[optional] save it

    The same idea as nms function, but added text visualization of confidence and bboxes
    """
    ans = img.copy()

    anchor = (0.063, 0.063)

    range_x, range_y = np.meshgrid(
        np.arange(19, dtype=pred.dtype),
        np.arange(19, dtype=pred.dtype),
    )

    X_shifted = range_x + pred[1]
    Y_shifted = range_y + pred[2]

    good_idxs = pred[0] > threshold

    labels = pred[0][good_idxs]

    xs = X_shifted[good_idxs]
    ys = Y_shifted[good_idxs]

    xs = (original_size*xs/19).astype(np.int32)
    ys = (original_size*ys/19).astype(np.int32)

    ws = (original_size*anchor[0]*np.exp(pred[3][good_idxs])).astype(np.uint8)
    hs = (original_size*anchor[1]*np.exp(pred[4][good_idxs])).astype(np.uint8)

    generator = zip(xs, ys, ws, hs)

    if nms_flag:
        _, def_generator = nms(pred, threshold=threshold,
                               visual=visual, original_size=original_size)
        generator = [i[1:] for i in def_generator]
        labels = [i[0] for i in def_generator]

    linewidth = max(round(sum(ans.shape) / 2 * 0.003), 2)
    tf = max(linewidth-1, 2)
    for idx, (x, y, w, h) in enumerate(generator):
        label = f"{labels[idx]:.3f}"
        lt = (int(x - w // 2), int(y - h // 2))
        br = (int(x + w // 2), int(y + h // 2))
        p1, p2 = lt, br
        color = (128, 128, 128)
        txt_color = (255, 255, 255)

        ans = cv2.rectangle(ans, lt, br, (255, 0, 0), 5)
        # text width, height
        w, h = cv2.getTextSize(
            label, 0, fontScale=linewidth / 3, thickness=tf)[0]
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(ans, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(ans, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, linewidth / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)

    if visual:
        plt.imshow(ans.astype(np.uint8))
        plt.show()

    if save:
        cv2.imwrite(fname, ans)

    return ans
