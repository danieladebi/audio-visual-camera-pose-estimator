torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', resolution=(256, 256), embedding_type='embeddings_slfm_egoexo4d', sound_size=1000)" \
    --test_dataset "EgoExo4D(split='val', resolution=(256, 256), embedding_type='embeddings_slfm_egoexo4d', sound_size=1000, seed=777)" \
    --model "DOACameraPoseModel(num_layers=40)" \
    --lr 1e-4 --min_lr 1e-6 --warmup_epochs 0 --epochs 100 --batch_size 256 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --output_dir "checkpoints/_egoexo4d-1000ms_slfm_embed_cp_"