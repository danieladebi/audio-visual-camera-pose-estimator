torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', use_slfm_and_doa=True, resolution=(256, 256),embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_doa=True)" \
    --test_dataset "EgoExo4D(split='val', use_slfm_and_doa=True, resolution=(256, 256),embedding_type='embeddings_doa_1000ms_alt', sound_size=1000, use_doa=True, seed=777)" \
    --model "DOACameraPoseModel(input_embed_dim=1384, num_layers=20)" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 0 --epochs 100 --batch_size 256 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_egoexo4d-1000ms_doa_cp_slfm_nocorruption_"