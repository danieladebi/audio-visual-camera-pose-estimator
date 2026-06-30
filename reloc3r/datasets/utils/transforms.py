# transform utilities adapted from DUSt3R
import torchvision.transforms as tvf
import torchvision.transforms.v2 as tvf2
from reloc3r.utils.image import ImgNorm

# Baseline (kept for backward compatibility)
ColorJitter = tvf.Compose([
    tvf.ColorJitter(0.5, 0.5, 0.5, 0.1),
    ImgNorm
])


# Evaluation (no augmentation)
Eval = tvf.Compose([
    ImgNorm
])

# Realistic image corruption transform using torchvision's RandomApply and GaussianBlur
Corrupted = tvf.Compose([
    tvf.RandomApply([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.3),
    tvf.RandomApply([tvf.GaussianBlur(kernel_size=(5, 9), sigma=1)], p=0.3), # 0,1,2,4, 8
    tvf.RandomApply([tvf.ToTensor(), tvf2.GaussianNoise(mean=0.0, sigma=0.2), tvf.ToPILImage()], p=0.2),
   # tvf.RandomApply([tvf.RandomEqualize()], p=0.3),
    ImgNorm
])