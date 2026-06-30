CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=(256, 256), is_audio=True, is_slfm=True)" \
    --test_dataset "SLfM_HM3D(split='train', resolution=(320,240), seed=777, sound_size=2550, use_audio=True)" \
    --batch_size 64 \

CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=(256, 256), is_audio=True, is_slfm=True)" \
    --test_dataset "SLfM_HM3D(split='val', resolution=(320,240), seed=777, sound_size=2550, use_audio=True)" \
    --batch_size 64 \

CUDA_VISIBLE_DEVICES=3 python eval_relpose.py \
    --model "Reloc3rRelpose(img_size=(256, 256), is_audio=True, is_slfm=True)" \
    --test_dataset "SLfM_HM3D(split='test', resolution=(320,240), seed=777, sound_size=2550, use_audio=True)" \
    --batch_size 64 \
   # --rot_only \
