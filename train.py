import os
import torch
import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from models import ArcNet
from models import YoLoss
from models import YoloEmb
from dataset import PILLfeats
from models import MobileNetV2Emb
from visual import map_visualizing
from utils import create_arc_dataset
from augmentations import train_det_ts
from dataset import DetectionExtractor
from dataset import CustomBatchSampler
from augmentations import train_arcface_tsfm
from augmentations import test_arcface_tsfm
from visual import arcface_feature_tsne_visual


def detection_backbone_train(description_csv="marked.csv", max_epochs=20):

    root_dir = "backbone_exp"
    idx = 0

    while os.path.exists(f"{root_dir}{idx}"):
        idx += 1

    root_dir = f"{root_dir}{idx}"
    os.makedirs(root_dir)

    # Crate datasets
    # Train dataset
    print("\nCreating training dataset...")
    create_arc_dataset(filename=description_csv, savedir="arc", train=True)
    # Validation dataset
    print("\nCreating validation dataset...")
    create_arc_dataset(filename=description_csv, savedir="arc", train=False)

    # Datasets
    train_csv_filename = "arcTrain.csv"
    val_csv_filename = "arcVal.csv"

    df_train = pd.read_csv(train_csv_filename)
    df_val = pd.read_csv(val_csv_filename)

    ds_train = PILLfeats(df_train, transforms=train_arcface_tsfm)
    ds_val = PILLfeats(df_val, transforms=test_arcface_tsfm)

    # Dataloaders
    dl_train = DataLoader(ds_train, num_workers=4,
                          batch_sampler=CustomBatchSampler(ds_train, 8))
    dl_val = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=4)

    # Model
    backbone_model = ArcNet(MobileNetV2Emb(), 9)

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epochs)

    # Training
    print("\nTraining started...")
    trainer.fit(backbone_model, dl_train, dl_val)

    # Saving model
    full_model_pth = os.path.join(root_dir, "BackboneWithArcFace.pth")
    embedding_model_pth = os.path.join(root_dir, "Backbone.pth")
    torch.save(backbone_model.state_dict(), full_model_pth)
    torch.save(backbone_model.emb_net.state_dict(), embedding_model_pth)

    # Visualization
    print("\nVisualizing...")
    arcface_feature_tsne_visual(model_path=embedding_model_pth, train_arc_ds_csv=train_csv_filename,
                                val_arc_ds_csv=val_csv_filename, save_fig_name=os.path.join(root_dir, "BackboneTSNEVisual.png"))

    return root_dir


def detection_train(backbone_path, max_epochs=300, train_ds_path="train", val_ds_path="val", freeze=False, lr_backbone=1e-4, cos_scheduler=True, description_csv="marked.csv"):

    # Creating experiments folder
    root_dir = "detection_exp"
    idx = 0

    while os.path.exists(f"{root_dir}{idx}"):
        idx += 1

    root_dir = f"{root_dir}{idx}"
    os.makedirs(root_dir)

    # Datsets
    print("\nCreating detection train dataset...")
    ds_tr = DetectionExtractor(os.path.join(train_ds_path, "images"), os.path.join(
        train_ds_path, "labels"), transforms=train_det_ts, rotate=True)

    print("\nCreating detection validation dataset...")
    ds_val = DetectionExtractor(os.path.join(
        val_ds_path, "images"), os.path.join(val_ds_path, "labels"), rotate=True)

    # Dataloaders
    dl_train = DataLoader(ds_tr, batch_size=4, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False)

    # Models
    mn = MobileNetV2Emb()
    if backbone_path is not None:
        print("\nLoading pretrained backbone...")
        mn.load_state_dict(torch.load(
            os.path.join(backbone_path, "Backbone.pth")))
    else:
        print("\nNo weights loaded. Training backbone with detector...")
    print("\nCreating detection model...")
    detection_model = YoloEmb(mn,
                              YoLoss(),
                              freeze=freeze,
                              lr_backbone=lr_backbone,
                              cos_scheduler=cos_scheduler,
                              saving_dir=root_dir).to('cuda')

    # Training
    print("\nTraining started...")
    trainer2 = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs)
    trainer2.fit(detection_model, dl_train, dl_val)

    # Saving model
    full_detection_model_pth = os.path.join(root_dir, "Detection_Last.pth")
    new_embedding_model_pth = os.path.join(root_dir, "TunedBackbone_Last.pth")
    torch.save(detection_model.state_dict(), full_detection_model_pth)
    torch.save(detection_model.emb_net.state_dict(), new_embedding_model_pth)

    # IoU history saving
    iou_history_array = np.array(detection_model.iou_list, dtype=object)
    np.save(os.path.join(root_dir, "iou_history.npy"), iou_history_array)

    # Visualizing for comparing
    new_best_embedding_model_pth = os.path.join(root_dir, "TunedBackbone_Best.pth")

    print("\nVisualizing...")
    map_visualizing(iou_history_array, detection_model, root_dir = root_dir, best_map_idx = detection_model.best_idx)

    visual_train_csv_filename = "arcTrain.csv"
    visual_val_csv_filename = "arcVal.csv"

    if not os.path.exists(visual_train_csv_filename) or not os.path.exists(visual_val_csv_filename):
        print("\nCSV files not found! Creating...")
        print("\nCreating training arc visualization dataset...")
        create_arc_dataset(filename = description_csv, savedir = "arc", train = True)
        # Validation dataset
        print("\nCreating validation arc visualization dataset...")
        create_arc_dataset(filename = description_csv,
                           savedir = "arc", train = False)

    # Last weights 
    arcface_feature_tsne_visual(model_path=new_embedding_model_pth, train_arc_ds_csv=visual_train_csv_filename,
                                val_arc_ds_csv=visual_val_csv_filename, save_fig_name=os.path.join(root_dir, "TunedBackboneTSNEVisual_Last.png"))
    # & best weights
    arcface_feature_tsne_visual(model_path=new_best_embedding_model_pth, train_arc_ds_csv=visual_train_csv_filename,
                                val_arc_ds_csv=visual_val_csv_filename, save_fig_name=os.path.join(root_dir, "TunedBackboneTSNEVisual_Best.png"))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default=None,
                        help="initial backbone weights(None => train with detector; Train => train backbone, then => detector")
    parser.add_argument('--epochs', type=int, default=300,
                        help="total detection model training epochs")
    parser.add_argument('--back_epochs', type=int, default=20,
                        help="total backbone training epochs")
    parser.add_argument('--description', type=str,
                        default="marked.csv", help="data labels csv")
    parser.add_argument('--train_path', type=str,
                        default="train", help="training data path")
    parser.add_argument('--val_path', type=str,
                        default="val", help="validation data path")
    parser.add_argument('--freeze', type=str,
                        default="False", help="freeze backbone training")
    parser.add_argument('--lr_backbone', type=float,
                        default=1e-4, help="backbone learning rate")
    parser.add_argument('--cos_scheduler', type=str,
                        default="True", help="Cosine scheduler / ReduceOnPlateau")

    return parser.parse_known_args()[0]


def main(opt):
    backbone_root = opt.backbone

    if opt.backbone is None:
        backbone_root = None
    elif opt.backbone == "Train":
        print(f"TRAINING BACKBONE MODEL...")
        backbone_root = detection_backbone_train(
            description_csv=opt.description, max_epochs=opt.back_epochs)

    freeze = False
    if opt.freeze != "False":
        freeze = True

    cos_scheduler = True
    if opt.cos_scheduler != "True":
        cos_scheduler = False

    print(f"TRAINING DETECTION MODEL...")
    detection_train(backbone_path=backbone_root, max_epochs=opt.epochs,
                    train_ds_path=opt.train_path, val_ds_path=opt.val_path, freeze=freeze, lr_backbone=opt.lr_backbone, cos_scheduler=cos_scheduler)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
