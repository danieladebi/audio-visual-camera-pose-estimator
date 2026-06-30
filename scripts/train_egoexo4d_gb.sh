torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', resolution=(256, 256))" \
    --test_dataset "EgoExo4D(split='val', resolution=(256, 256), seed=777)" \
    --model "Reloc3rRelposeJointModel()" \
    --pretrained "checkpoints/_egoexo4d-gradient_blending-512_/checkpoint-best.pth" \
    --lr 1e-5 --min_lr 1e-6 --warmup_epochs 0 --epochs 100 --batch_size 32 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --freeze_encoder \
    --output_dir "checkpoints/_egoexo4d-gradient_blending-512_"