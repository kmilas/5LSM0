wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 200 \
    --lr 0.00003 \
    --crop_size 448 \
    --resume True \
    --model-name eomt \
    --eval-freq 10 \
    --num-workers 10 \
    --seed 42 \
    --clip-grad True \
    --experiment-id "eomt-segm-adamW-warmup-polylr-3e-5-crop-448-batch-8-iters-60k-gradclip" \