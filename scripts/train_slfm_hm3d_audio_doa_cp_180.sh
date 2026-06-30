torchrun --nproc_per_node=8 train.py \
    --train_dataset "SLfM_HM3D(split='train', resolution=(256, 256), sound_size=2550, use_audio=True, use_doa=True)" \
    --test_dataset "SLfM_HM3D(split='val', resolution=(256, 256), sound_size=2550, use_audio=True, use_doa=True, seed=777)" \
    --model "DOACameraPoseModel(input_embed_dim=180, num_layers=20)" \
    --lr 1e-4 --min_lr 1e-7 --warmup_epochs 0 --epochs 100 --batch_size 256 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_slfm_hm3d_doa_cp_180"