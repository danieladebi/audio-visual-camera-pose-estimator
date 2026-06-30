CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "DOACameraPoseModel()" \
    --test_dataset "EgoExo4D(split='train', resolution=(256, 256),embedding_type='embeddings_doa_60ms', sound_size=60, use_doa=True, seed=777)" \
    --batch_size 256 \


# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
#     --model "DOACameraPoseModel()" \
#     --test_dataset "EgoExo4D(split='val', resolution=(256, 256),embedding_type='embeddings_doa_60ms', sound_size=60, use_doa=True, seed=777)" \
#     --batch_size 256 \