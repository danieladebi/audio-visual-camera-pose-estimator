torchrun --nproc_per_node=8 train.py \
    --train_dataset "10000 @ SLfM_HM3D(split='train', resolution=(320, 240), use_img=True, embedding_type='doa', seed=777, transform=ColorJitter)" \
    --test_dataset "1000 @ SLfM_HM3D(split='val', resolution=(320, 240), use_img=True, embedding_type='doa', seed=777)" \
    --model "Reloc3rRelpose(img_size=(320, 240), has_audio=True, has_audio_embedding=True)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 32 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --freeze_encoder \
    --output_dir "checkpoints/_slfm_hm3d-vision_doa_embed-512_"