CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=(512, 64))" \
    --test_dataset "EgoExo4D(split='train', resolution=(512,64), seed=777)" \
    --batch_size 64 \
   # --rot_only \
