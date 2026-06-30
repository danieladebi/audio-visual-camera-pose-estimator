torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', use_slfm_and_doa=True, resolution=(256, 256),embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_img=True, transform=ColorJitter)" \
    --test_dataset "EgoExo4D(split='val', use_slfm_and_doa=True, resolution=(256, 256),embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_img=True, seed=777)" \
    --model "Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True, use_doa_and_embed=True)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 32 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --freeze_encoder \
    --output_dir "checkpoints/_egoexo4d-vision_slfm_embed_and_doa_360_1000ms-512_"