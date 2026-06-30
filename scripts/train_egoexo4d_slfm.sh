torchrun --nproc_per_node=8 train.py \
    --train_dataset "50000 @ EgoExo4D(split='train', resolution=(256, 256), sound_size=1000, use_img=True, use_slfm=True)" \
    --test_dataset "10000 @ EgoExo4D(split='val', resolution=(256, 256), seed=777, sound_size=1000, use_img=True, use_slfm=True)" \
    --model "SLfMNet()" \
    --lr 1e-4 --min_lr 1e-6 --warmup_epochs 0 --epochs 100 --batch_size 128 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_egoexo4d-slfm_no_vision_"