CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "PolicyClassificationModel()" \
    --test_dataset "EgoExo4D(split='val', resolution=(512, 96), sound_size=1000, use_yamnet_embed=True, use_policy_model=True, seed=777)" \
    --batch_size 256 \