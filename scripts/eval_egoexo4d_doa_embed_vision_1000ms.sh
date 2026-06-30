CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=256, has_audio=True, has_audio_embedding=True)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256,256), embedding_type='embeddings_doa_1000ms', sound_size=1000, seed=777, use_img=True)" \
    --batch_size 256 \
   # --rot_only \
