torchrun --nproc_per_node=8 train.py \
    --train_dataset "EgoExo4D(split='train', use_policy_model=True, sound_size=60, resolution=(512, 24))" \
    --test_dataset "EgoExo4D(split='val', use_policy_model=True, sound_size=60, resolution=(512, 24), seed=777)" \
    --model "PolicyClassificationModel()" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 0 --epochs 200 --batch_size 128 --accum_iter 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --train_criterion "PolicyBCELoss()" \
    --test_criterion "PolicyBCELoss()" \
    --output_dir "checkpoints/_egoexo4d-policy_oursvsdoaandslfm_60ms_0-0_errs_audioweighted10x_200epochs_"