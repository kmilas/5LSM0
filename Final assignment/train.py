"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
  
from torch.optim import AdamW, SGD
from timm.scheduler import PolyLRScheduler
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.transforms.v2 import (
    Compose,
    RandomRotation,
    RandomApply,
    RandomCrop,
    RandomHorizontalFlip,
    RandomPhotometricDistort,
    ColorJitter,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    ToTensor,
    RandomResizedCrop
)
from torch.amp import GradScaler
from unet import UNet
from model import PSPNet, EPSPNet
from vit_model import Segmenter
from uper_head import Dinov2Uper
from eomt_segm import SegmenterEoMT

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch PSPNet model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--crop_size", type=int, default=512, help="Crop size augmentations")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from the last checkpoint")
    parser.add_argument("--model-name", type=str, default='dinov2s_uper', help="Select model")
    parser.add_argument("--freeze", type=bool, default=False, help="Use freeze backbone")
    parser.add_argument("--eval-freq", type=int, default=10, help="After how many epoch log mIoU")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # True: clip-grad = 0 False: clip-grad = 1 
    parser.add_argument("--clip-grad", type=int, default=0, help="Clip gradients")
    parser.add_argument("--scheduler", type=int, default=0, help="If scheduler==0 PolyLR, if scheduler==1 StepLR ")
    parser.add_argument("--experiment-id", type=str, default="pspnet-training", help="Experiment ID for Weights & Biases")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ImageNet statistics
    imgnet_1k_mean = [0.485, 0.456, 0.406]
    imgnet_1k_std = [0.229, 0.224, 0.225]
  
    train_transform = Compose([
        ToImage(),
        Resize(size=(int(1024), int(2048))),  # Apply random scale
        RandomCrop((args.crop_size, args.crop_size)),
        RandomHorizontalFlip(p=0.5),
        RandomPhotometricDistort(p=0.5),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=imgnet_1k_mean, std=imgnet_1k_std), # cityscape statistics


    ])

    val_transform = Compose([
            ToImage(),
            Resize((args.crop_size,args.crop_size)),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=imgnet_1k_mean, std=imgnet_1k_std),])


    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=train_transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=val_transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    miou_metric = MulticlassJaccardIndex(num_classes=19, average="macro", ignore_index=255).to(device)

    max_iters = len(train_dataloader)*args.epochs
    
    iter = 0
 
    print('Resume:',args.resume)
    print('Crop size:',args.crop_size)

    if(args.model_name == 'dinov2s_uper'):

        model = Dinov2Uper(n_cls=19, patch_size=14, d_encoder=384, freeze=args.freeze)

    elif(args.model_name == 'dinov2s_linear'):

        model = Segmenter(
            n_cls=19, 
            patch_size=14, 
            d_encoder=384, 
            backbone='dinov2_vits14', 
            depth_anything=False
        )


    elif(args.model_name == 'dinov2b_linear'):
        # 'dinov2_vitb14_reg'
        # base 786
        # small 384
        model = Segmenter(
            n_cls=19, 
            patch_size=14, 
            d_encoder=768, 
            backbone='dinov2_vitb14_reg', 
            depth_anything=False
        )
        #if args.resume:
        #    dinov2e5_100 = 'checkpoints/dinov2b-linear-adamW-step-lr-1e-5-crop-560-batch-8/final_model-epoch=0099-val_loss=0.1707789365734373.pth'
        #    ckpt = torch.load(
        #            dinov2e5_100,
        #            map_location='cpu', 
        #            weights_only=True
        #    )
        #    model.load_state_dict(ckpt)
    elif(args.model_name == 'depth_anything_linear'):
        model = Segmenter(
            n_cls=19, 
            patch_size=14, 
            d_encoder=768, 
            backbone='dinov2_vitb14', 
            depth_anything=True
        )

    elif (args.model_name == 'eomt'):
        model = SegmenterEoMT(
            num_classes=19, 
            patch_size=14, 
            d_model=768, 
            image_size=(args.crop_size, args.crop_size)
        )

    else:
        raise KeyError(f'invalid {args.model_name} for model_name')

    model.to(device)
    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class
    if(args.model_name == 'dinov2s_uper'):
        aux_criterion = nn.CrossEntropyLoss(ignore_index=255)


    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.scheduler == 1:
        lr_scheduler = PolyLRScheduler(optimizer, t_initial=max_iters, warmup_t=1500, warmup_lr_init=args.lr*1e-6, warmup_prefix=True)
    elif args.scheduler == 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    else: 
        raise KeyError(f'invalid {args.scheduler} for scheduler') 
    #optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.05)
    
    #optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    #optimizer = torch.optim.SGD([
    #    {'params': model.backbone.parameters(), 'lr': 0.1*args.lr},
    #    {'params': list(model.ppm.parameters()) + list(model.decoder.parameters()), 'lr': args.lr}], 
    #    momentum=0.9, weight_decay=1e-4)

    # pspnet poly learning rate
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                        lr_lambda=lambda step: (1 - step / max_iters) ** 0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scaler = GradScaler()
    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        epoch_start = True
        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if(args.model_name == 'dinov2s_uper'):
                    outputs, aux_logits = model(images)
                    loss = criterion(outputs, labels) + 0.4*aux_criterion(aux_logits, labels)
                else:
                    # im_size only for dinov2 linear
                    outputs = model(images, im_size=(args.crop_size,args.crop_size))
                    loss = criterion(outputs, labels) 

            scaler.scale(loss).backward()
            if args.clip_grad==0:
                scaler.unscale_(optimizer)  # Important for proper grad clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)

            scaler.step(optimizer)
            scaler.update()

            
            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            if args.scheduler == 1:
                lr_scheduler.step(iter)
            iter += 1
        if args.scheduler == 0:
            scheduler.step()    
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                # im_size only for dinov2 linear
                outputs = model(images, im_size=(args.crop_size,args.crop_size))
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                if (epoch+1)%10 == 0:
                    preds = torch.argmax(outputs, dim=1)

                    # ignore_index = 255
                    valid_mask = labels != 255
                    labels_valid = torch.where(valid_mask, labels, torch.tensor(0, device=labels.device))
                    preds_valid = torch.where(valid_mask, preds, torch.tensor(0, device=preds.device))

                    assert preds_valid.min() >= 0 and preds_valid.max() < 19, f"preds out of range: {preds_valid.min()} to {preds_valid.max()}"
                    assert labels_valid.min() >= 0 and labels_valid.max() < 19, f"labels out of range: {labels_valid.min()} to {labels_valid.max()}"
                    miou_metric.update(preds_valid, labels_valid)
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            if (epoch+1)%10 == 0:
                miou = miou_metric.compute()
                wandb.log({
                    "mIoU": miou
                }, step=(epoch + 1) * len(train_dataloader) - 1)

            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)

    
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
