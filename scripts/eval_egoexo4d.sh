# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
#     --model "Reloc3rRelpose(img_size=256)" \
#     --test_dataset "EgoExo4D(split='train', resolution=(256,256), seed=777, use_img=True)" \
#     --batch_size 256 \


CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=256)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256,256), seed=777, use_img=True, transform=ImgNorm)" \
    --batch_size 256 \
