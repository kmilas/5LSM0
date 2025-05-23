wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.00001 \
    --crop_size 560 \
    --model-name dinov2b_linear \
    --eval-freq 10 \
    --num-workers 10 \
    --scheduler 0 \
    --seed 42 \
    --clip-grad 1 \
    --experiment-id "best-dinov2b-linear-adamW-steplr-1e-5-crop-560-batch-8" \