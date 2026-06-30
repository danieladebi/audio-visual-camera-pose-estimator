torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', resolution=(256, 256),embedding_type='embeddings_doa_60ms', sound_size=60, use_doa=True, transform=ColorJitter)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256, 256),embedding_type='embeddings_doa_60ms', sound_size=60, use_doa=True, seed=777)" \
    --model "DOACameraPoseModel(num_layers=20)" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 0 --epochs 100 --batch_size 256 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_egoexo4d-60ms_doa_cp_"