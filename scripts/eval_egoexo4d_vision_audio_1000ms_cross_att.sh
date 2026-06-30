CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=256)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256, 256),sound_size=1000, use_img=True, embedding_type='audio', seed=777)" \
    --batch_size 256 \
   # --rot_only \
