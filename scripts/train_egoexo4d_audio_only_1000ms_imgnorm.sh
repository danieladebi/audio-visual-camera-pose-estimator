torchrun --nproc_per_node=8 train.py \
    --train_dataset "50000 @ EgoExo4D(split='train', binaural=True, resolution=(512, 96), sound_size=1000, use_audio=True)" \
    --test_dataset "10000 @ EgoExo4D(split='val', binaural=True, resolution=(512, 96), seed=777, sound_size=1000, use_audio=True)" \
    --model "Reloc3rRelpose(img_size=(512, 96), in_channels=2, is_audio=True)" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 16 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_egoexo4d-audio_only_1000ms_specnorm-512_"