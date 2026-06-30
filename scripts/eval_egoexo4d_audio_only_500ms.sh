# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
#     --model "Reloc3rRelpose(img_size=(512, 48))" \
#     --test_dataset "EgoExo4D(split='train', resolution=(512, 48), seed=777, sound_size=500, use_audio=True)" \
#     --batch_size 256 \

CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=(512, 48))" \
    --test_dataset "EgoExo4D(split='val', resolution=(512, 48), seed=777, sound_size=500, use_audio=True)" \
    --batch_size 256 \
   # --rot_only \
