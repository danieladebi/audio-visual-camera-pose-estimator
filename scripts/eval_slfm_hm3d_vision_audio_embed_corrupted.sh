CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=(320, 240))" \
    --test_dataset "SLfM_HM3D(split='test', resolution=(320,240), seed=777, use_img=True, embedding_type='audio', transform=Corrupted)" \
    --batch_size 64 \
   # --rot_only \
