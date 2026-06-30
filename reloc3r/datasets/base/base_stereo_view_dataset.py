# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# base class for implementing datasets
# --------------------------------------------------------
import PIL
import numpy as np
import torch


from reloc3r.datasets.utils.transforms import ImgNorm
import reloc3r.datasets.utils.cropping as cropping
from .easy_dataset import EasyDataset
from pdb import set_trace as bb


class BaseStereoViewDataset (EasyDataset):
    """ Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(self, *,  # only keyword arguments
                 split=None,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=ImgNorm,
                 aug_crop=False,
                 seed=None):
        self.num_views = 2
        self.split = split
        self._set_resolutions(resolution)

        self.transform = transform
        if isinstance(transform, str):
            transform = eval(transform)

        self.aug_crop = aug_crop
        self.seed = seed

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = '['+';'.join(f'{w}x{h}' for w, h in self._resolutions)+']'
        return f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace('self.', '').replace('\n', '').replace('   ', '')

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views

        # check data-types
        for v, view in enumerate(views):
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            # width, height = view['img'].size #if len(view['img']) == 2 else 192, 128
            # view['true_shape'] = np.int32((height, width))
            # #if len(view['img']) == 2:
            # view['img'] = self.transform(view['img'])
            # encode the audio spectrogram
            if 'img' in view and view['img'] is not None and not isinstance(view['img'].size, int):
                if 'slfm' in view['dataset'].lower():
                    width, height = view['img'].shape[-1], view['img'].shape[-2] # if len(view['img']) == 2 else 256, 256
                    view['true_shape'] = np.int32((height, width))
                else:
                    width, height = view['img'].size # if len(view['img']) == 2 else 256, 256
                    view['true_shape'] = np.int32((height, width))
                    view['img'] = self.transform(view['img'])
                    # DEBUG: visualize, save, then stop
                    

                    # tensor = view['img']
                    # if torch.is_tensor(tensor):
                    #     x = tensor.detach().cpu()
                    #     if x.ndim == 3:
                    #         # keep first 3 channels if more
                    #         if x.shape[0] > 3:
                    #             x = x[:3]
                    #         # map (C,H,W) -> (H,W,C)
                    #         x = x.permute(1, 2, 0).contiguous()
                    #     x = x.float()
                    #     # normalize to 0..255 for saving
                    #     x_min, x_max = float(x.min()), float(x.max())
                    #     if x_max > x_min:
                    #         x = (x - x_min) / (x_max - x_min)
                    #     x = (x * 255).clamp(0, 255).to(torch.uint8).numpy()
                    # else:
                    #     # assume PIL image
                    #     x = np.array(tensor)
                    #     if x.ndim == 2:
                    #         x = np.repeat(x[..., None], 3, axis=2)
                    #     elif x.shape[2] > 3:
                    #         x = x[..., :3]

                    # pil_img = PIL.Image.fromarray(x)
                    # import os 
                    # os.makedirs("corrupted", exist_ok=True)
                    # debug_path = f"./frame_pairs/{view['take_name']}/{view['label']}_{idx}_{v}.jpg"
                    # pil_img.save(debug_path, format="JPEG")
                    # # raise Exception("corruption")

            else:
                if 'img' not in view or view['img'] is None: 
                    pass
                else:
                    if "doas" in view:
                        height, width = view["audio_spec"].shape[-2:]
                        view['true_shape'] = np.int32((height, width))
                    else: 
                        height, width = view['img'].shape[-2:] #if len(view['img']) == 2 else 192, 128
                        view['true_shape'] = np.int32((height, width))
                #if len(view['img']) == 2:            

            # assert 'camera_intrinsics' in view

            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            
            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            if 'img' in view and view['img'] is not None and not "slfm" in view['dataset'].lower():
                transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
        return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            assert width >= height
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(self, image, intrinsics, resolution, rng=None, info=None):
        """ 
        siyan: this function can change the camera center, but the corresponding pose does not transform accordingly...
        """
        if not isinstance(image, PIL.Image.Image): 
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        # image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        image, intrinsics = cropping.crop_image(image, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1*W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        # image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)
        image, intrinsics = cropping.rescale_image(image, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        # image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        image, intrinsics2 = cropping.crop_image(image, intrinsics, crop_bbox)

        return image, intrinsics2


def is_good_type(key, v):
    """ returns (is_good, err_msg) 
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view['dataset'])
    label = sel(view['label'])
    instance = sel(view['instance'])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view['true_shape']
    if width < height:
        # rectify portrait to landscape
        if 'img' in view:
            assert view['img'].shape == (3, height, width) or view['img'].shape == (2, height, width)  or view['img'].shape == (7, height, width)
            view['img'] = view['img'].swapaxes(1, 2)

        if 'audio_spec' in view:
            assert view['audio_spec'].shape == (2, height, width) or view['audio_spec'].shape == (7, height, width)
           # view['audio_spec'] = view['audio_spec'].swapaxes(-2, -1)
            return 
        
        K = view['camera_intrinsics']
        K[0,0], K[1,1] = K[1,1], K[0,0]
        K[0,2], K[1,2] = K[1,2], K[0,2]
        view['camera_intrinsics'] = K

