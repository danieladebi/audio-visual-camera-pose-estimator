CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True, use_doa_and_embed=True)" \
    --test_dataset "EgoExo4D(split='val', use_imu=True, resolution=(256, 256),embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_img=True, seed=777)" \
    --batch_size 256 \
   # --rot_only \
