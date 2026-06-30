CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "AudioDOAModel(embed_dim=1024, model_type=50)" \
    --test_dataset "SLfM_HM3D_DOA(split='train',sound_size=2550, resolution=(256,256), seed=777)" \
    --batch_size 256 \
   
CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "AudioDOAModel(embed_dim=1024, model_type=50)" \
    --test_dataset "SLfM_HM3D_DOA(split='val', sound_size=2550, resolution=(256,256), seed=777)" \
    --batch_size 256 \

CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "AudioDOAModel(embed_dim=1024, model_type=50)" \
    --test_dataset "SLfM_HM3D_DOA(split='test',sound_size=2550, resolution=(256,256), seed=777)" \
    --batch_size 256 \