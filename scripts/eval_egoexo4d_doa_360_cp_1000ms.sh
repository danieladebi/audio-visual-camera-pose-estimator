CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "DOACameraPoseModel(input_embed_dim=360)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256, 256), sound_size=1000, use_doa_only_model=True, seed=777)" \
    --batch_size 256 \