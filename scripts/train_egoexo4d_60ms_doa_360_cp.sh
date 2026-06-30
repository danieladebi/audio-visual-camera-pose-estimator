torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', resolution=(256, 256), sound_size=60, use_doa_only_model=True)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256, 256), sound_size=60, use_doa_only_model=True, seed=777)" \
    --model "DOACameraPoseModel(input_embed_dim=360)" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 256 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_egoexo4d-60ms_doa_360_cp_"