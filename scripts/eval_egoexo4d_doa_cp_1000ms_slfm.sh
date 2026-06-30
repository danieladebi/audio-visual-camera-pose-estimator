CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "DOACameraPoseModel()" \
    --test_dataset "EgoExo4D(split='train', use_slfm_and_doa=True, resolution=(256, 256),embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_doa=True, seed=777)" \
    --batch_size 256 \

# CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
#     --model "DOACameraPoseModel()" \
#     --test_dataset "EgoExo4D(split='val',  use_slfm_and_doa=True, resolution=(256, 256),embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_doa=True, seed=777)" \
#     --batch_size 256 \