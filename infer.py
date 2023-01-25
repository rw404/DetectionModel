import os
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from utils import nms
from models import YoLoss
from models import YoloEmb
from utils import load_image
from models import MobileNetV2Emb
from dataset import DetectionInference
from utils import convert_to_original



def inference(img_path, idx, saving_dir, model: YoloEmb, visual=False):

    # Creating special dataset without labels
    img_ds = DetectionInference(img_path=img_path)
    img_dl = DataLoader(img_ds, batch_size=4, shuffle=False)

    # Dataloader consists of 4 images(4 rotations of original)
    img = next(iter(img_dl))

    # Convert img to CPU device
    pred = model(img.cpu())
    output = model.infer_test(pred.detach().cpu().numpy())

    # Normalized predictions
    _, pred_list = nms(output[0], original_size=2000)

    # Optional visualiziation
    if visual:
        plot_img, _, _ = load_image(img_path)
        plot_img = cv2.cvtColor(plot_img.copy(), cv2.COLOR_BGR2RGB)

        convert_to_original(
            output[0],
            plot_img.copy().astype(np.uint8),
            save=True, nms_flag=True, fname=saving_dir+'/'+str(idx)+".jpg",
            original_size=2000,
        )

    return pred_list


def run_folder(descriptions="marked.csv", model_path="./pretrained/Detector.pth"):
    """Visualizing model result

    descriptions - csv with "img" column for model inference
    model_path - path for detection model weights
    """

    # Original image directory
    root_dir = "./inference_input"
    idx = 0

    while os.path.exists(f"{root_dir}{idx}"):
        idx += 1

    root_dir = f"{root_dir}{idx}"
    os.makedirs(root_dir)

    # Predictions image directory
    pred_dir = "./inference_pred"
    pred_idx = 0

    while os.path.exists(f"{pred_dir}{pred_idx}"):
        pred_idx += 1

    pred_dir = f"{pred_dir}{pred_idx}"
    os.makedirs(pred_dir)

    pd_description = pd.read_csv(descriptions)

    # List of input images paths
    path_list = []

    for row_idx in range(pd_description.shape[0]):
        img_path = pd_description["img"][row_idx]

        cur_img = cv2.imread(img_path)

        new_path = os.path.join(root_dir, f"{row_idx}.jpg")

        new_img = Image.fromarray(cur_img)
        new_img.save(new_path)

        path_list.append(new_path)

    # Model loading
    model = YoloEmb(MobileNetV2Emb(), YoLoss())

    model.load_state_dict(torch.load(model_path))

    model.eval()
    model.to("cpu")

    # Run model
    for idx, path in tqdm(enumerate(path_list), total=len(path_list)):
        predictions = inference(path, idx, pred_dir, model, visual=True)

        # Save predictions(normalized by original size)
        with open(os.path.join(pred_dir, f"{idx}.txt"), "w") as f:
            for pred in predictions:
                f.write(f"{pred[0]} {pred[1]} {pred[2]} {pred[3]} {pred[4]}\n")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default="./pretrained/Detector.pth",
                        help="pretrained model weight's path")
    parser.add_argument('--description', type=str,
                        default="marked.csv", help="data labels csv")

    return parser.parse_known_args()[0]

def main(opt):
    run_folder(descriptions=opt.description, model_path=opt.weight_path)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)