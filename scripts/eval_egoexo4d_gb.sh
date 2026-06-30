CUDA_VISIBLE_DEVICES=0\3 python eval_relpose.py \
    --model "Reloc3rRelposeJointModel()" \
    --test_dataset "EgoExo4D(split='val', resolution=(256,256), seed=777)" \
    --batch_size 32 \
   # --rot_only \
