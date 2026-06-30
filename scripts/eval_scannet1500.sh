# Running evaluation for scannet1500

CUDA_VISIBLE_DEVICES=0 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=512)" \
    --test_dataset "ScanNet1500(resolution=(512,384), seed=777)" \
    --batch_size 32
   # --test_dataset "EgoExo4D(resolution=(256,256), seed=777)" \

