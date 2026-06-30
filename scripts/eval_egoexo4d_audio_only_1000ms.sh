# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
#     --model "Reloc3rRelpose(img_size=(512, 96))" \
#     --test_dataset "EgoExo4D(split='train', resolution=(512, 96), seed=777, sound_size=1000, use_audio=True)" \
#     --batch_size 64 \

CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=(512, 96))" \
    --test_dataset "EgoExo4D(split='val', resolution=(512, 96), seed=777, sound_size=1000, use_audio=True)" \
    --batch_size 64 \
   # --rot_only \
