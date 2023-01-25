import os
import math
import torch
import torchvision
import numpy as np
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

from utils import rot_90
from utils import nms

# ---- BACKBONE SECTION ----
""" ARCFACE SECTION

Models for backbone training with pytorch lightning

- ArcLoss as default metric learning model
- ArcNet:
    - emb_net as backbone(ResNet18Emb / MobileNetV2Emb)
    - linearizer for classifier input
    - classifier(ArcLoss) - fully-connected network for metric training
- ResNet18Emb / MobileNetV2Emb - main networks for detection model usage
"""


class ArcLoss(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcLoss, self).__init__()
        self.s = 30.0
        self.m = 0.4
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False)
        )
        self.eps = 1e-7

    def forward(self, x, labels):
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        cos_theta = self.fc(x)

        numerator = cos_theta.transpose(0, 1)
        numerator = torch.diagonal(numerator[labels])
        numerator = torch.clamp(numerator, -1+self.eps, 1-self.eps)
        numerator = self.s*torch.cos(torch.acos(numerator)+self.m)

        excluded_real_class = torch.cat([
            torch.cat([cos_theta[i, :y], cos_theta[i, y+1:]])[None, :]
            for i, y in enumerate(labels)
        ], dim=0)
        denominator = torch.exp(
            numerator) + torch.sum(torch.exp(self.s*excluded_real_class), dim=1)

        loss = numerator - torch.log(denominator)
        return cos_theta, -loss.mean()


class ArcNet(pl.LightningModule):
    def __init__(self, emb_net, train_classes):
        super().__init__()
        self.emb_net = emb_net
        self.linearizer = torch.nn.Flatten(start_dim=1)
        self.classifier = ArcLoss(self.emb_net.out_channels, train_classes)

    def forward(self, x):
        output = self.emb_net(x)
        output = self.linearizer(output)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        outputs = self(x)
        outputs, loss = self.classifier(outputs, y)
        acc = (outputs.argmax(dim=1) == y).sum().item() / len(outputs)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        outputs = self(x)
        outputs, loss = self.classifier(outputs, y)
        acc = (outputs.argmax(dim=1) == y).sum().item() / len(outputs)
        self.log('val_loss', loss, prog_bar=True)
        self.log('vall_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.emb_net.parameters(), "lr": 1e-3},
            {"params": self.classifier.parameters(), "lr": 1e-5},
        ], lr=1e-5)
        return optimizer


class ResNet18Emb(nn.Module):
    """ArcFace backbone model

    basic architecture - resnet18 without classifier subnet
    additional layers - conv from feature space 512 x 2 x 2 to 512 x 1 x 1
    """

    def __init__(self):
        super(ResNet18Emb, self).__init__()
        backbone = torchvision.models.resnet18()
        feats = torch.nn.ModuleList(backbone.children())[:-2]

        self.net = nn.Sequential(
            *feats,
            torch.nn.Conv2d(512, 512, 2)
        )
        self.out_channels = 512

    def forward(self, x):
        output = self.net(x)

        return output


class MobileNetV2Emb(nn.Module):
    """ArcFace backbone model

    basic architecture - mobilenet_v3_small without classifier subnet
    additional layers - conv from feature space 1024 x 2 x 2 to 512 x 1 x 1
    """

    def __init__(self):
        super(MobileNetV2Emb, self).__init__()
        backbone = torchvision.models.mobilenet_v3_small()
        feats = nn.ModuleList(backbone.children())[0]

        self.net = nn.Sequential(
            *feats,
            nn.Conv2d(feats[-1].out_channels, 512, 2)
        )
        self.out_channels = 512

    def forward(self, x):
        output = self.net(x)

        return output

# ---- DETECTION SECTION ----


class YoloEmb(pl.LightningModule):
    """Yolo-like detection model

    - emb_net - backbone network(MobileNetV2Emb was used)
    - freeze flag - boolean flag for freezing trained backbone

    - bbox_head - additional network(bbox extractor)
    """

    def __init__(self, emb_net, loss, freeze: bool = True, num_classes=1,
                 anchors=(
                     (0.063, 0.063),
                 ),
                 lr_backbone=1e-4,
                 cos_scheduler=True,
                 saving_dir=None
                 ):
        super().__init__()
        self.emb_net = emb_net
        self.freeze = freeze

        for p in self.emb_net.parameters():
            p.requires_grad = not self.freeze

        self.bbox_head = nn.Sequential(
            nn.BatchNorm2d(self.emb_net.out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.emb_net.out_channels,
                      128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 5, 1)
        )

        self.register_buffer("anchors", torch.tensor(anchors))
        self.train_thr = 0.2

        self.loss = loss

        y1 = 1.
        y2 = 0.2

        # Cosine lr scheduler
        self.cos_scheduler = cos_scheduler
        self.decay_rule = lambda x: (
            (1 - math.cos(x * math.pi / 300)) / 2) * (y2 - y1) + y1

        # Logging list of iou lists per epoch(validation data) for visualization
        self.iou_list = []

        # Backbone learning rate
        self.lr_backbone = lr_backbone

        # For best model weights saving
        self.max_map95 = 0.
        self.saving_best = saving_dir
        self.best_idx = 0

    def forward(self, x, features: bool = True):
        output = self.emb_net(x)
        output = self.bbox_head(output)

        nB, _, nH, nW = output.shape
        output = output.view(nB, -1, nH, nW).permute(0, 2, 3, 1)

        output = torch.cat([
            output[:, None, :, :, 4].sigmoid(),  # Confidence
            output[:, None, :, :, 0].sigmoid(),  # X center
            output[:, None, :, :, 1].sigmoid(),  # Y center
            output[:, None, :, :, 2],            # Width
            output[:, None, :, :, 3],            # Height
        ], 1)

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch

        outputs = self(x)

        loss = self.loss(outputs, y)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        outputs = self(x)

        loss = self.loss(outputs, y)
        ious = self.get_iou_tensor(outputs.detach().cpu(
        ).numpy(), y.detach().cpu().numpy()).sum(dim=0)
        inter = ious[0]
        union = ious[1]
        iou = 0 if union == 0 else inter/union

        # MAP calculation
        inference_prediction = self.infer_test(outputs.detach().cpu().numpy())
        _, nms_predictions = nms(inference_prediction[0])
        _, nms_targets = nms(y[0].detach().cpu().numpy())

        current_iou = self.correct_iou(nms_predictions, nms_targets)

        return {"val_loss": loss, "val_iou": iou, "val_iou_box": current_iou}

    def validation_epoch_end(self, outputs):
        """log and display average val metrics"""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_iou = torch.stack([x["val_iou"] for x in outputs]).mean()

        # Extract iou from each batch
        map_iou = []
        for x in outputs:
            for iou in x['val_iou_box']:
                map_iou.append(iou)

        # Logging iou progression for visualization
        self.iou_list.append(map_iou)

        # MaP@.5
        map5 = self.get_map(np.array(map_iou))

        # MaP@.5-.95
        map95 = self.get_map(
            np.array(map_iou), threshold=np.linspace(0.5, 0.95, 10))

        print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.5f} Val_iou(entire image): {avg_iou:.5f} MaP@.5: {map5:.5f} MaP@.5-.95: {map95:.5f}", end=" ")

        if map95 > self.max_map95:
            self.max_map95 = map95
            self.best_idx = self.trainer.current_epoch
            if self.saving_best is not None:
                full_best_detection_model_pth = os.path.join(
                    self.saving_best, "Detection_Best.pth")
                new_best_embedding_model_pth = os.path.join(
                    self.saving_best, "TunedBackbone_Best.pth")
                torch.save(self.state_dict(), full_best_detection_model_pth)
                torch.save(self.emb_net.state_dict(),
                           new_best_embedding_model_pth)

        self.log('val_loss', avg_loss, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('val_iou_img', avg_iou, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('val_map@.5', map5, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('val_map@.5-.95', map95, prog_bar=True,
                 on_epoch=True, on_step=False)

    # MAP SECTION
    def iou_wh(self, pred, target):
        """Calculate iou between to lists: [conf, x_c, y_c, w, h]

        pred - predicted list
        target - correct list
        """
        left_up_pred = (max(0, (pred[1]-pred[3]/2)), max(0, pred[2]-pred[4]/2))
        left_up_target = (
            max(0, target[1]-target[3]/2), max(0, target[2]-target[4]/2))

        right_down_pred = (
            min(1, pred[1]+pred[3]/2), min(1, pred[2]+pred[4]/2))
        right_down_target = (
            min(1, target[1]+target[3]/2), min(1, target[2]+target[4]/2))

        left_up = (max(left_up_pred[0], left_up_target[0]), max(
            left_up_pred[1], left_up_target[1]))
        right_down = (min(right_down_pred[0], right_down_target[0]), min(
            right_down_pred[1], right_down_target[1]))

        if left_up[0] > right_down[0] or left_up[1] > right_down[1]:
            return 0.0

        intersect = (right_down[0]-left_up[0])*(right_down[1]-left_up[1])

        union = pred[3]*pred[4]+target[3]*target[4]-intersect

        return intersect/union

    def correct_iou(self, pred_list, target_list):
        """Match prediction feature map with targets and return iou list

        pred_list as list of lists [conf, x_c, y_c, w, h]
        target_list same

        returns iou between matched predictions and targets
        """
        pred_iou = []
        used_idxs = []

        for item in pred_list:
            cmpr_list = []
            for cmpr in target_list:
                cur_iou = self.iou_wh(np.array(item)/640, np.array(cmpr)/640)
                if cur_iou < 0 or cur_iou > 1.0:
                    cur_iou = 0
                cmpr_list.append(cur_iou)

            cmpr_list = np.array(cmpr_list)
            matched_idx = np.argmax(cmpr_list)

            # Different predictions for different targets
            # and different targets for different predictions
            #assert matched_idx not in used_idxs, "TERMINATION USED IDX!!!!"

            pred_iou.append(cmpr_list.max())
            used_idxs.append(matched_idx)

        return pred_iou

    def get_map(self, iou, threshold=np.array([0.5]), visual=False):
        """MaP calculation

        iou - list of ious
        threshold - array of thresholds:
            - np.array([0.5]) for MaP@.5, 
            - np.linspace(0.5, 0.95, 10) for MaP@.5-.95
        visual - boolean flag(FOR MaP@.5 only!!!):
            - if True then return precision and recall for plotting
        """
        ap_thr = []

        for thr in threshold:
            tp = iou > thr

            fpc = (1 - tp).cumsum(0)
            tpc = tp.cumsum(0)

            recall = tpc / iou.shape[0]
            precision = tpc / (tpc + fpc)

            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([1.0], precision, [0.0]))

            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

            x = np.linspace(0, 1, 101)
            ap = np.trapz(np.interp(x, mrec, mpre), x)

            ap_thr.append(ap)

            if visual:
                return ap, mrec, mpre

        return np.array(ap_thr).mean()
    # END OF MAP SECTION

    def get_iou_tensor(self, pred, target, visual: bool = False):
        nB = pred.shape[0]

        ans = torch.zeros((nB, 2))

        for i in range(nB//4):
            # List of 4 rotated imgs(0, 90, 180, 270)
            confs = []
            for rotated in range(4*i, 4*i+4):
                # Rotate feature map and correct x, y, w, h
                rotated_res = rot_90(pred[rotated], k=4-rotated % 4)
                confs.append(rotated_res)

            # Find rotations to extract max confidence
            # we need only confidence layer
            conf_stacked = np.stack(confs, axis=0)
            conf_idx = conf_stacked[:, 0].argmax(axis=0)

            # Construct new result feature map
            result_map = np.zeros(pred[4*i].shape, dtype=pred[4*i].dtype)
            for rotation_idx in range(4):
                # chose coords with max confidence at feature map x_ in [0, 19), y in [0, 19)
                x_, y_ = np.where(conf_idx == rotation_idx)
                result_map[:, x_, y_] = conf_stacked[rotation_idx, :, x_, y_].T

            # NMS of msx confidence matrix
            pred_feats, _ = nms(result_map, visual=visual)
            target_feats, _ = nms(target[4*i], visual=visual)

            ans[i, 0] = (pred_feats & target_feats).sum()
            ans[i, 1] = (pred_feats | target_feats).sum()

        return ans

    def infer_test(self, pred):
        """ Inference method

        Excpect DetectionExtractor dataset with train flag as TRUE,
        because model runs with batch of 4 rotated images, 
        then reverse these rotations
        and using non-maximum suppression for prediction
        """
        nB = pred.shape[0]

        ans = []

        for i in range(nB//4):
            # List of 4 rotated imgs(0, 90, 180, 270)
            confs = []
            for rotated in range(4*i, 4*i+4):
                # Rotate feature map and correct x, y, w, h
                rotated_res = rot_90(pred[rotated], k=4-rotated % 4)
                confs.append(rotated_res)
                ###cur_preds, _ = nms(pred[rotated], visual = visual)

            # Find rotations to extract max confidence
            # we need only confidence layer
            conf_stacked = np.stack(confs, axis=0)
            conf_idx = conf_stacked[:, 0].argmax(axis=0)

            # Construct new result feature map
            result_map = np.zeros(pred[4*i].shape, dtype=pred[4*i].dtype)
            for rotation_idx in range(4):
                # chose coords with max confidence at feature map x_ in [0, 19), y in [0, 19)
                x_, y_ = np.where(conf_idx == rotation_idx)
                result_map[:, x_, y_] = conf_stacked[rotation_idx, :, x_, y_].T

            # NMS of msx confidence matrix
            ans.append(result_map)

        return np.stack(ans, axis=0)

    def configure_optimizers(self):
        if self.freeze:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        else:
            optimizer = torch.optim.Adam(
                [
                    {"params": self.emb_net.parameters(), "lr": self.lr_backbone,
                     "weight_decay": 0},
                    {"params": self.bbox_head.parameters(), "lr": 1e-2,
                     "weight_decay": 0},
                ],
                lr=1e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='max',
                                                                  factor=0.5,
                                                                  patience=20,
                                                                  verbose=True)

        lr_dict = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.decay_rule) if self.cos_scheduler else lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_map@.5-.95",
            "strict": True,
            "name": None,
        }

        return [optimizer], [lr_dict]


class YoLoss(nn.Module):
    """Weighted loss for detection model

    box    - weight of detection parameters loss(width, height)
    center - weight of detection coordinates loss
    obj    - weight of confidence layer loss
    """

    def __init__(
        self,
        box=0.5,
        center=0.5,
        obj=1.0,
    ):
        super().__init__()

        self.obj = float(obj)
        self.box = float(box)
        self.center = float(center)

        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, x, y):
        nB = x.shape[0]

        loss_box = self.loss(x[:, 3:], y[:, 3:])
        loss_center = self.loss(x[:, 1:3], y[:, 1:3])
        loss_obj = self.loss(x[:, 0], y[:, 0])

        return (self.obj * loss_obj + self.box * loss_box + self.center * loss_center) / nB
