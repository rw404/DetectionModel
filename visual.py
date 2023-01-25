import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data.dataloader import DataLoader

from models import YoloEmb
from dataset import PILLfeats
from models import MobileNetV2Emb
from dataset import CustomBatchSampler

def plot_2d_data(data, labels, title='Исходные данные', cmap='tab20', ax=None, save_fig_name="BackboneTSNEVisual.png"):
    """2d scatter plot. 
    :param np.ndarray data
    :param Union[list, np.ndarray] labels
    :param str title
    :param str cmap
    :param ax Optional[matplotlib.axes.Axes]
        If ax is None, then creates a new one and saving
        Overwise graph adds to exsisting ax
    """
    n_clusters = len(np.unique(labels))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = None
        
    scatter = ax.scatter(
        data[:, 0], data[:, 1], c=labels, 
        cmap=plt.get_cmap(cmap, n_clusters)
    )

    cbar = plt.colorbar(scatter, label='PILL cls', ax=ax)
    cbar.set_ticks(np.min(labels) + (np.arange(n_clusters) + 0.5) * (n_clusters - 1) / n_clusters)
    cbar.set_ticklabels(np.unique(labels))

    ax.set_title(title)
    ax.grid(True)
    
    if fig is not None:
        fig.tight_layout()
        plt.savefig(save_fig_name)

def arcface_feature_tsne_visual(model_path = "backbone.pth", title = "Backbone visual", train_arc_ds_csv = "arcTrain.csv", val_arc_ds_csv = "arcVal.csv", save_fig_name="BackboneTSNEVisual.png"):
    """Backbone feature dimension reducing visualization
    
    loads model from model_path,
    creates TSNE with perplexity = 200,
    visualising it with input title and saving
    """
    test_tsfm = A.Compose(
        []
    )
    
    df_train = pd.read_csv(train_arc_ds_csv)
    df_val = pd.read_csv(val_arc_ds_csv)

    TSNE_DS = PILLfeats(pd.concat([df_train, df_val]).iloc[:25000], transforms=test_tsfm)
    dl = DataLoader(TSNE_DS, num_workers = 4, batch_sampler=CustomBatchSampler(TSNE_DS, 8))

    mn = MobileNetV2Emb()
    mn.load_state_dict(torch.load(model_path))

    feats = torch.empty(0)
    labels = torch.empty(0)
    for imgs, cur_labels in tqdm(dl):
        cur_feats = mn(imgs).detach().cpu().view(-1, 512)
        feats = torch.cat([feats, cur_feats], dim = 0)
        labels = torch.cat([labels, cur_labels], dim = 0)

    plot_2d_data(TSNE(perplexity = 200).fit_transform(feats.numpy()), labels.numpy(), title = title, cmap = 'tab20b', save_fig_name = save_fig_name)

def map_visualizing(iou_list, model: YoloEmb, root_dir = "./", map5_plot_fname = "MaP@05_Best.png", 
                    map5_history_fname = "MaP@05Progress.png", map95_history_fname = "MaP@05-095Progress.png", best_map_idx = -1):
    """Plot and visual result MaP and MaP progress per epoch
    """
    # MaP@.5 & MaP@.5-.95 metric
    last_epoch_iou = np.array(iou_list[best_map_idx])
    
    map05, mrec, mpre = model.get_map(last_epoch_iou, visual=True)
    map095 = model.get_map(last_epoch_iou, threshold=np.linspace(0.5, 0.95, 10))

    # Result plot
    plt.figure(figsize=(10, 10))
    plt.title(f"MaP@.5")
    plt.plot(mrec, mpre, label = f"MaP@.5={map05:.3f}; MaP@.5-.95={map095:.4f}")
    plt.legend()
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, map5_plot_fname))

    # Progress visualization
    epochs = np.arange(iou_list.shape[0])
    map5_progress = []
    map95_progress = []

    for epoch in range(iou_list.shape[0]):
        epoch_map5 = model.get_map(np.array(iou_list[epoch]))
        epoch_map95 = model.get_map(np.array(iou_list[epoch]), threshold=np.linspace(0.5, 0.95, 10))

        map5_progress.append(epoch_map5)
        map95_progress.append(epoch_map95)
    
    # MaP@.5 progress
    plt.figure(figsize=(10, 10))
    plt.title(f"MaP@.5 Progress")
    plt.plot(epochs, map5_progress, label = f"MaP@.5 per epoch")
    plt.legend()
    plt.ylabel("MaP@.5")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, map5_history_fname))

    # MaP@.5-.95 progress
    plt.figure(figsize=(10, 10))
    plt.title(f"MaP@.5-.95 Progress")
    plt.plot(epochs, map95_progress, label = f"MaP@.5-.95 per epoch")
    plt.legend()
    plt.ylabel("MaP@.5-.95")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, map95_history_fname))