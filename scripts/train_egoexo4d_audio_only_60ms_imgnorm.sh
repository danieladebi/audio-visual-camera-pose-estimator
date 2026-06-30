torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', resolution=(512, 16), sound_size=60, use_audio=True)" \
    --test_dataset "EgoExo4D(split='val', resolution=(512, 16), seed=777, sound_size=60, use_audio=True)" \
    --model "Reloc3rRelpose(img_size=(512, 16), is_audio=True)" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 64 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_egoexo4d-audio_only_60ms_imgnorm-512_"