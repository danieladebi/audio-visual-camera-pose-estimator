torchrun --nproc_per_node=8 train.py \
    --train_dataset "50000 @ EgoExo4D(split='train', resolution=(256, 256), use_img=True)" \
    --test_dataset "15000 @ EgoExo4D(split='val', resolution=(256, 256), use_img=True, seed=777, transform=Corrupted)" \
    --model "Reloc3rRelpose(img_size=256, has_audio=False, has_audio_embedding=False)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 0 --epochs 100 --batch_size 64 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --freeze_encoder \
    --output_dir "checkpoints/_egoexo4d-corrupted_vision_test-512_"