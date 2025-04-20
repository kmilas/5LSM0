
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from vit_model import Segmenter
from uper_head import Dinov2Uper
import torch
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.segmentation import DiceScore
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
)

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Set test_time to 'ms' if you want multi scale testing
test_time = '' # NO dinov2_uper only depth_any and dinov2 !!! set it to ''

def pad_to_multiple(image, divisor=14):
    B, C, H, W = image.shape
    pad_H = (divisor - H % divisor) % divisor
    pad_W = (divisor - W % divisor) % divisor
    padding = (0, pad_W, 0, pad_H)  # (left, right, top, bottom)
    return F.pad(image, padding, mode='reflect'), pad_H, pad_W

def unpad(image, pad_H, pad_W):
    if pad_H > 0:
        image = image[:, :, :-pad_H, :]
    if pad_W > 0:
        image = image[:, :, :, :-pad_W]
    return image

# do not use it for dinov2_uper
def multi_scale_inference(model, image, scales=[0.5, 0.75, 1.0, 1.25, 1.5], num_classes=19):

    B, C, H, W = image.shape
    device = image.device
    logits_accum = torch.zeros((B, num_classes, H, W), device=device)

    for scale in scales:
        new_h, new_w = int(H * scale), int(W * scale)
        scaled_img = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

        padded_img, pad_H, pad_W = pad_to_multiple(scaled_img, divisor=14)
        _, _, padded_H, padded_W = padded_img.shape
        output = model(padded_img, im_size=(padded_H, padded_W))
        output = unpad(output, pad_H, pad_W)

        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)

        logits_accum += output

    logits_avg = logits_accum / len(scales)

    return logits_avg

# If dinov2s_uper variants image size 672 else 560
image_size = 560

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

imgnet_1k_mean = [0.485, 0.456, 0.406]
imgnet_1k_std = [0.229, 0.224, 0.225]
data_dir = './data/cityscapes'

# If you want to run the robustness tests leave it else set it to ''
mode = ''

if mode == 'robustness':
    albumentations_transform = A.Compose([
        A.OneOf([
            A.RandomRain(p=1),
            A.RandomFog(p=1),
            A.RandomSnow(p=1),
        ], p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), contrast_limit=0.3, p=1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=30, p=1),
            A.ISONoise(p=1),
        ], p=0.5),

        A.MotionBlur(blur_limit=3, p=0.2),
        A.GaussNoise(p=0.2),


        A.Resize(image_size, image_size),
        A.Normalize(mean=imgnet_1k_mean, std=imgnet_1k_std),
        ToTensorV2(), 
    ])
    city_dataset = Cityscapes(
        data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=None
    )
    class CityscapesAlbumentationsWrapper(Dataset):
        def __init__(self, cityscapes_dataset, transform=None):
            self.dataset = cityscapes_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]

            # Convert PIL to numpy arrays for albumentations
            image = np.array(image)
            label = np.array(label)

            if self.transform:
                transformed = self.transform(image=image, mask=label)
                image = transformed['image']
                label = transformed['mask']

            return image, label

    valid_dataset = CityscapesAlbumentationsWrapper(city_dataset, transform=albumentations_transform)
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=0
    )

else:
    val_transform = Compose([
                ToImage(),
                Resize((image_size, image_size)),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=imgnet_1k_mean, std=imgnet_1k_std),])
    valid_dataset = Cityscapes(
        data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=val_transform
    )
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=10
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dinov2e5_50 = 'checkpoints/dinov2b-linear-adamW-step-lr-1e-5-crop-560-batch-8/best_model-epoch=0050-val_loss=0.1657727017762169.pth'
dinov2e5_100 = 'checkpoints/dinov2b-linear-adamW-step-lr-1e-5-crop-560-batch-8/final_model-epoch=0099-val_loss=0.1707789365734373.pth'
dinov2e4_55 = 'checkpoints/dinov2b-linear-adamW-step-lr-1e-4-crop-560-batch-8/best_model-epoch=0055-val_loss=0.18928335323220208.pth'
dinov2e4_100 = 'checkpoints/dinov2b-linear-adamW-step-lr-1e-4-crop-560-batch-8/final_model-epoch=0099-val_loss=0.196692104495707.pth'
depth_any_e5_27 = 'checkpoints/depth-anything-linear-adamW-steplr-1e-5-crop-560-batch-8/best_model-epoch=0027-val_loss=0.16759864051663687.pth'
dinov2s_uper_32 = 'checkpoints/dinov2-uper-aux-4e-1-adamW-steplr-3e-5-crop-672-batch-8/best_model-epoch=0032-val_loss=0.16852520642772553.pth'
dinov2s_uper_50 = 'checkpoints/dinov2-uper-aux-4e-1-adamW-steplr-3e-5-crop-672-batch-8/final_model-epoch=0049-val_loss=0.17037812608575065.pth'
dinov2s_uper_32_dropout = 'checkpoints/dinov2-uper-dropout-aux-4e-1-adamW-steplr-3e-5-crop-672-batch-8/best_model-epoch=0032-val_loss=0.16742488611785192.pth'
dinov2s_uper_50_dropout = 'checkpoints/dinov2-uper-dropout-aux-4e-1-adamW-steplr-3e-5-crop-672-batch-8/final_model-epoch=0049-val_loss=0.16938070645408024.pth'
ckpt = torch.load(
    dinov2e5_50,
    map_location='cpu', 
    weights_only=True
)

# Select model accordingly
model_name = ''

if model_name == 'dinov2_uper':
    model = Dinov2Uper(n_cls=19, patch_size=14, d_encoder=384)
    image_size = 672
elif model_name == 'depth_any':
    model = Segmenter(
        n_cls=19, 
        patch_size=14, 
        d_encoder=768, 
        backbone='dinov2_vitb14', 
        depth_anything=True
    )
    image_size = 560
else:
    # depth is only true when you use depth_any_e5_27 model
    depth = True
    backbone = 'dinov2_vitb14'
    model = Segmenter(
        n_cls=19, 
        patch_size=14, 
        d_encoder=768, 
        backbone='dinov2_vitb14_reg', 
        depth_anything=False
    )
    image_size = 560

#print(ckpt.keys()) , strict=False
model.load_state_dict(ckpt)
model.to(device)



# Initialize Metric meters
miou_metric = MulticlassJaccardIndex(num_classes=19, average="macro", ignore_index=255).to(device)
dice = DiceScore(
    num_classes=19,          
    include_background=False, # Exclude background (class 0) if needed
    average='none',          
    input_format='index',     # You're using integer label maps
    zero_division=0.0         
).to(device)


model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(valid_dataloader):

        labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
        images, labels = images.to(device), labels.to(device)
        labels = labels.long().squeeze(1)  # Remove channel dimension

        if test_time == 'ms':
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
            outputs = multi_scale_inference(model, images, scales=scales)
        else:
            outputs = model(images)

        preds = torch.argmax(outputs, dim=1)


        
        # ignore_index = 255
        valid_mask = labels != 255
        labels_valid = torch.where(valid_mask, labels, torch.tensor(0, device=labels.device))
        preds_valid = torch.where(valid_mask, preds, torch.tensor(0, device=preds.device))

        assert preds_valid.min() >= 0 and preds_valid.max() < 19, f"preds out of range: {preds_valid.min()} to {preds_valid.max()}"
        assert labels_valid.min() >= 0 and labels_valid.max() < 19, f"labels out of range: {labels_valid.min()} to {labels_valid.max()}"
        miou_metric.update(preds_valid, labels_valid)
        dice.update(preds_valid, labels_valid)

miou = miou_metric.compute()
print(f"Mean IoU: {miou:.4f}")

class_dice_scores = dice.compute()
mean_dice_score = torch.mean(class_dice_scores)


for i, score in enumerate(class_dice_scores):
    print(f"Class {i}: Dice Score = {score:.4f}")
print(f"\nMean Dice Score: {mean_dice_score:.4f}")
