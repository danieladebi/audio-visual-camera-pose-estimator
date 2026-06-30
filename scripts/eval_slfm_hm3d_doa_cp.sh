CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "DOACameraPoseModel(input_embed_dim=360, num_layers=20)" \
    --test_dataset "SLfM_HM3D(split='test', resolution=(256, 256), sound_size=2550, use_audio=True, use_doa=True, seed=777)" \
    --batch_size 256 \