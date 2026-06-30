torchrun --nproc_per_node=8 train.py \
    --train_dataset "SLfM_HM3D(split='train', resolution=(256, 256), sound_size=2550, use_audio=True)" \
    --test_dataset "SLfM_HM3D(split='val', resolution=(256, 256), seed=777, sound_size=2550, use_audio=True)" \
    --model "Reloc3rRelpose(img_size=(256, 256), is_audio=True, is_slfm=True)" \
    --lr 1e-5 --min_lr 1e-6 --warmup_epochs 5 --epochs 100 --batch_size 28 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_slfm_hm3d-audio_only_norm-512_"