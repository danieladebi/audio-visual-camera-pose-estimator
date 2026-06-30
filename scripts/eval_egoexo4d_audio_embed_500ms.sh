CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=256)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256,256),  embedding_type='embeddings_500ms', sound_size=500, seed=777, use_img=True)" \
    --batch_size 64 \
   # --rot_only \
