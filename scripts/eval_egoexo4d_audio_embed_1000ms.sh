CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=256)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256,256), embedding_type='embeddings_1000ms', sound_size=1000, use_img=True)" \
    --batch_size 128 \
   # --rot_only \
