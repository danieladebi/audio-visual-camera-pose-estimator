torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', resolution=(512, 48), sound_size=500, use_audio=True)" \
    --test_dataset "EgoExo4D(split='val', resolution=(512, 48), seed=777, sound_size=500, use_audio=True)" \
    --model "Reloc3rRelpose(img_size=(512, 48), is_audio=True)" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 64 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_egoexo4d-audio_only_500ms-512_"