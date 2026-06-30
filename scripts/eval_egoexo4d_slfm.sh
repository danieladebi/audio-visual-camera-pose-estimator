# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
#     --model "SLfMNet()" \
#     --test_dataset "EgoExo4D(split='train', resolution=(256,256), seed=777, use_img=True, use_slfm=True)" \
#     --batch_size 128 \


CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "SLfMNet()" \
    --test_dataset "EgoExo4D(split='val', resolution=(256,256), seed=777, use_img=True, use_slfm=True)" \
    --batch_size 128 \
   --rot_only \
