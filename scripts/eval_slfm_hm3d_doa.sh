CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "AudioDOAModel(embed_dim=1024, model_type=50)" \
    --test_dataset "SLfM_HM3D_DOA(split='train', resolution=(320,240), seed=777, sound_size=2550)" \
    --batch_size 64 \

CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "AudioDOAModel(embed_dim=1024, model_type=50)" \
    --test_dataset "SLfM_HM3D_DOA(split='val', resolution=(320,240), seed=777, sound_size=2550)" \
    --batch_size 64 \

CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "AudioDOAModel(embed_dim=1024, model_type=50)" \
    --test_dataset "SLfM_HM3D_DOA(split='test', resolution=(320,240), seed=777, sound_size=2550)" \
    --batch_size 64 \
   # --rot_only \
