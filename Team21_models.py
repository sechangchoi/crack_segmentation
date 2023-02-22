import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
import segmentation_models_pytorch as smp # https://github.com/qubvel/segmentation_models.pytorch



class Team21_Model(pl.LightningModule):

    accuracy = 0
    recall = 0
    precision = 0
    f1_score_value = 0
    
    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        result = torch.sigmoid(mask)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch[0]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        mask = batch[1]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        global accuracy
        global f1_score_value
        global precision
        global recall
        
        ########
        f1_score_value = torchmetrics.functional.f1_score(torch.sigmoid(logits_mask), mask.long())
        precision = torchmetrics.functional.precision(logits_mask.view(-1), mask.long().view(-1))
        recall = torchmetrics.functional.recall(logits_mask.view(-1), mask.long().view(-1))
        accuracy = torchmetrics.functional.accuracy(logits_mask.view(-1), mask.long().view(-1))
        ########


        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # iou_score
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # loss
        total_loss = 0
        for index in range(len(outputs)):
            total_loss += outputs[index]['loss']
        
        total_loss /= len(outputs)
        

        metrics = {
            f"{stage}_Accuracy": accuracy,
            f"{stage}_loss": total_loss,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_Precision": precision,
            f"{stage}_Recall": recall,
            f"{stage}_f1_score": f1_score_value
            }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)