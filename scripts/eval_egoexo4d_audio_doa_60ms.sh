CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "AudioDOAModel(embed_dim=1024, model_type=50)" \
    --test_dataset "EgoExo4D_DOA(split='train',sound_size=60, resolution=(512,24), seed=777)" \
    --batch_size 256 \
   # --rot_only \
CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "AudioDOAModel(embed_dim=1024, model_type=50)" \
    --test_dataset "EgoExo4D_DOA(split='val', sound_size=60, resolution=(512,24), seed=777)" \
    --batch_size 256 \