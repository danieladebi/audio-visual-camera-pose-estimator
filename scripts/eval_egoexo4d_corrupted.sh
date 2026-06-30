CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=256)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256,256), use_img=True, seed=777, transform=Corrupted)" \
    --batch_size 128 \
   # --rot_only \
