torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D_DOA(split='train', sound_size=1000, resolution=(256, 188), n_src=1)" \
    --test_dataset "EgoExo4D_DOA(split='val', sound_size=1000, resolution=(256, 188),  n_src=1)" \
    --model "AudioDOAModel(embed_dim=1024, model_type=50, n_degs=360)" \
    --lr 1e-4 --min_lr 1e-6 --warmup_epochs 0 --epochs 100 --batch_size 128 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --train_criterion "DOALoss()" \
    --test_criterion "DOALoss()" \
    --output_dir "checkpoints/_egoexo4d-audio_doa_1000ms_alt_"
    