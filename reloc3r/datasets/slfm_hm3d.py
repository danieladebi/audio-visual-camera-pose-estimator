import numpy as np

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2
from reloc3r.utils.image import ImgNorm
import os
import torch
import torchvision.transforms as tvf
import torchvision.transforms.v2 as tvf2

import scipy
from scipy.signal import fftconvolve

from glob import glob
import matplotlib

import pyroomacoustics as pra
import soundfile as sf
import json
import PIL 
# mics = {
#     "mic0": [0.0340384, -0.0896267, -0.070348],
#     "mic1": [-0.0062721, -0.0523816, -0.0304851],
#     "mic2": [0.0311063, -0.0167887, -0.0145339],
#     "mic3": [-0.0093178, -0.0021818, -0.0068198],
#     "mic4": [-0.0070644, -0.098121, -0.0720965],
#     "mic5": [-0.0016655, 0.0629508, -0.0809149],
#     "mic6": [0.002788, -0.0478431, -0.1658549]
# }
# L = np.array([
#             [mics[mic][0] for mic in mics],
#             [mics[mic][1] for mic in mics],
#             [mics[mic][2] for mic in mics]
#         ])
mics = {
    "mic0": [-0.08, 0],
    "mic1": [0.08, 0]
}
locs = np.array([
            [mics[mic][0] for mic in mics],
            [mics[mic][1] for mic in mics],
        ])

DATA_ROOT = "/vision/vision_data_2/SLfM/Dataset/AI-Habitat/ProcessedData/hm3d-2view-rotation"
SOUND_ROOT = "/vision/vision_data_2/SLfM/Dataset/AI-Habitat/LibriSpeech/ProcessedData"

class SLfM_HM3D(BaseStereoViewDataset):
    
    def __init__(self, use_img=False, use_doa=False, use_doa_only_model=False, use_audio=False, embedding_type=None, sound_size=2550, n_src=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT

        self.use_img = use_img 
        self.use_doa = use_doa or use_doa_only_model
        self.use_doa_only_model = use_doa_only_model
        self.use_audio = use_audio or use_doa or use_doa_only_model
        self.embedding_type = embedding_type
        self.sound_size = sound_size
        self.sampling_rate = 16000
        self.n_src = n_src

        # filter angle
        self.pairs_path = f"/vision/vision_data_2/SLfM/slfm/data/AI-Habitat/data-split/hm3d-4view-rotation/{self.split}.csv" # filter_angle
        self.source_sounds_path = f"/vision/vision_data_2/SLfM/Dataset/AI-Habitat/LibriSpeech/data-split/LibriSpeech/{self.split}.csv"
        with open(self.pairs_path, "rb") as f:
            lines = f.read().splitlines()
            self.pairs_list = [
                eval(line.strip())
                for line in lines
                if line.strip() and eval(line.strip()) != "path"
            ]
        updated_list = []
        for pair_path in self.pairs_list:
            pair_sub_folders = pair_path.split("rotation/")[-1]
            pair_path = os.path.join(self.data_root, pair_sub_folders)
            if os.path.exists(pair_path):
                updated_list.append(pair_path)
        self.pairs_list = updated_list

        with open(self.source_sounds_path, "rb") as f:
            lines = f.read().splitlines()
            self.source_sounds_list = [
                eval(line.strip())
                for line in lines
                if line.strip() and eval(line.strip()) != "path"
            ]
        self.rng = np.random.default_rng(2022)

    def normalize_audio(self, samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        samples[samples > 1.] = 1.
        samples[samples < -1.] = -1.
        return samples 

    def normalize_magnitude(self, spec):
        # import pdb; pdb.set_trace()
        spec_min = -100
        spec_max = 60

        spec = torch.maximum(spec, torch.tensor(1e-5))
        spec = 20 * torch.log10(spec)
        spec = (spec - spec_min) / (spec_max - spec_min) * 2 - 1
        spec = torch.clip(spec, -1.0, 1.0)

        return spec

    def read_audio(self, audio_path, start=0, stop=None):
        # import pdb; pdb.set_trace()
        audio, audio_rate = sf.read(audio_path, start=start, stop=stop, dtype='float32', always_2d=True)
        # repeat in case audio is too short
        if not stop == None:
            desired_audio_length = int(stop - start)
            if audio.shape[0] < desired_audio_length:
                repeat_times = np.ceil(desired_audio_length / audio.shape[0])
                audio = np.tile(audio, (int(repeat_times), 1))[:desired_audio_length, :]

        if audio_rate != self.sampling_rate:
            audio = scipy.signal.resample(audio, int(audio.shape[0] / audio_rate * self.sampling_rate), axis=0)
            audio_rate = self.sampling_rate
        return audio, audio_rate

    def __len__(self):
        return len(self.pairs_list)

    def _get_views(self, idx, resolution, rng):
        views = []
        pair_path = self.pairs_list[idx]
        pair_sub_folders = pair_path.split("rotation/")[-1]
        pair_path = os.path.join(self.data_root, pair_sub_folders)
        metadata_path = os.path.join(pair_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        source_label = f'{"-".join(pair_sub_folders.split("/"))}_0'
        target_label = f'{"-".join(pair_sub_folders.split("/"))}_1'

        source_rotation = metadata["camera_0_angle"]
        target_rotation = metadata["camera_1_angle"]

        source_spec = None
        target_spec = None
        source_azimuths = None
        target_azimuths = None


        source_rotation = np.deg2rad(source_rotation)  # convert degrees to radians if needed
        source_R = np.array([
            [np.cos(source_rotation), -np.sin(source_rotation), 0],
            [np.sin(source_rotation),  np.cos(source_rotation), 0],
            [0, 0, 1]
        ])
        source_translation = np.array([0, 0, 0])  # assuming source is at origin

        target_rotation = np.deg2rad(target_rotation)  # convert degrees to radians if needed
        target_R = np.array([
            [np.cos(target_rotation), -np.sin(target_rotation), 0],
            [np.sin(target_rotation),  np.cos(target_rotation), 0],
            [0, 0, 1]
        ])
        target_translation = np.array([0, 0, 0])  # assuming target

        source_pose = np.concatenate([source_R, source_translation[:, None]], axis=1)  # 3x4
        target_pose = np.concatenate([target_R, target_translation[:, None]], axis=1)  # 3x4

        source_pose = np.concatenate([source_pose, np.array([[0, 0, 0, 1]])], axis=0)  # 4x4
        target_pose = np.concatenate([target_pose, np.array([[0, 0, 0, 1]])], axis=0)  # 4x4

        if self.use_audio:
            source_binaural_rir_path = os.path.join(pair_path, "binaural_rirs", "sound_0_camera_0_rir.wav")
            target_binaural_rir_path = os.path.join(pair_path, "binaural_rirs", "sound_0_camera_1_rir.wav")

            source_binaural_rir, _ = sf.read(source_binaural_rir_path, dtype='float32', always_2d=True)
            target_binaural_rir, _ = sf.read(target_binaural_rir_path, dtype='float32', always_2d=True)

            mono_source_sounds_path = self.rng.choice(self.source_sounds_list, 1, replace=False)[0]
            mono_source_sounds_sub_folders = mono_source_sounds_path.split("ProcessedData/")[-1]
            mono_source_sounds_path = os.path.join(SOUND_ROOT, mono_source_sounds_sub_folders)

            with open(os.path.join(mono_source_sounds_path, 'meta.json'), "r") as f:
                source_meta = json.load(f)
            audio_rate = source_meta['audio_sample_rate']
            audio_length = source_meta['audio_length']
            clip_length = np.rint(self.sound_size/1000 * audio_rate).astype(int)
            remain_length = int(audio_length - self.sound_size/1000)
            if self.split == 'train' and remain_length > 0:
                start = int(self.rng.choice(remain_length) * audio_rate)
            else:
                start = 0

            mono_source_sound, _ = self.read_audio(os.path.join(mono_source_sounds_path, 'audio.wav'), start=start, stop=start+clip_length)
            mono_source_sound = mono_source_sound.mean(-1)
            desired_rms = 0.06 * self.rng.random() + 0.07
            mono_source_sound = self.normalize_audio(mono_source_sound, desired_rms=desired_rms)

            audio_length = mono_source_sound.shape[0]
            source_binaural_convolved = np.array([fftconvolve(mono_source_sound, source_binaural_rir[:, channel]) for channel in range(source_binaural_rir.shape[-1])])
            source_binaural_convolved = source_binaural_convolved[:, :audio_length]
            source_render_audio = source_binaural_convolved
            target_binaural_rir_convolved = np.array([fftconvolve(mono_source_sound, target_binaural_rir[:, channel]) for channel in range(target_binaural_rir.shape[-1])])
            target_binaural_rir_convolved = target_binaural_rir_convolved[:, :audio_length]
            target_render_audio = target_binaural_rir_convolved
            del mono_source_sound


            # compute spectrogram
            C, L = source_render_audio.shape

            frames = 256
            hop_length = int(L // (frames - 1))

            n_fft = 512
            source_spec = torch.stft(
                input=torch.tensor(source_render_audio),
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=400,
                return_complex=True
            )

            target_spec = torch.stft(
                input=torch.tensor(target_render_audio),
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=400,
                return_complex=True
            )

            # Begin mag + phase
            # source_spec = torch.view_as_real(source_spec)
            # source_spec = source_spec.permute(0, 3, 1, 2)
            # source_spec = source_spec.contiguous().view(-1, *source_spec.shape[2:])

            # target_spec = torch.view_as_real(target_spec)
            # target_spec = target_spec.permute(0, 3, 1, 2)
            # target_spec = target_spec.contiguous().view(-1, *target_spec.shape[2:])
            # End mag + phase

            if self.use_doa:
                source_doa = pra.doa.normmusic.NormMUSIC(
                            L=locs,
                            n_src=self.n_src,
                            fs=self.sampling_rate,
                            nfft=n_fft,
                           # azimuth=np.linspace(-90.,90.,180)*np.pi/180,
                        )
                source_X = pra.transform.stft.analysis(source_render_audio.T, L=n_fft, hop=hop_length)
                source_X = source_X.transpose(2, 1, 0)  # shape: (n_mics, n_frames, n_freq_bins)
          
                source_doa.locate_sources(source_X, num_src=1)
                source_azimuths = source_doa.grid.values.copy()
                target_doa = pra.doa.normmusic.NormMUSIC(
                            L=locs,
                            n_src=self.n_src,
                            fs=self.sampling_rate,
                            nfft=n_fft,
                          #  azimuth=np.linspace(-90.,90.,180)*np.pi/180,
                        )

                target_X = pra.transform.stft.analysis(target_render_audio.T, L=n_fft, hop=hop_length)
                target_X = target_X.transpose(2, 1, 0)  # shape: (n_mics, n_frames, n_freq_bins)
                target_doa.locate_sources(target_X, num_src=1)
                target_azimuths = target_doa.grid.values.copy()

                source_azimuths = np.array(source_azimuths).astype(np.float32)  # in degrees
                target_azimuths = np.array(target_azimuths).astype(np.float32)  # in degrees
                source_azimuths = (source_azimuths - source_azimuths.min()) / (source_azimuths.max() - source_azimuths.min()) 
                target_azimuths = (target_azimuths - target_azimuths.min()) / (target_azimuths.max() - target_azimuths.min())                
     
            del source_render_audio, target_render_audio

            source_spec = source_spec.contiguous() #.view(C, *source_spec.shape[1:])
            target_spec = target_spec.contiguous() #.view(C, *target_spec.shape[1:])

            source_spec = torch.abs(source_spec)
            target_spec = torch.abs(target_spec)

            source_spec = self.normalize_magnitude(source_spec)[:, :, :]
            target_spec = self.normalize_magnitude(target_spec)[:, :, :]

            source_spec = source_spec[:, :-1, :]
            target_spec = target_spec[:, :-1, :]

            SpecNorm = tvf.Compose([tvf.Normalize((0.5, 0.5), (0.5, 0.5))])

            C, H, W = source_spec.shape
            source_spec = SpecNorm(source_spec).numpy()
            target_spec = SpecNorm(target_spec).numpy()

            source_spec = source_spec.astype(np.float32).reshape(C,H,W)
            target_spec = target_spec.astype(np.float32).reshape(C,H,W)
        

        source_color_img = None
        target_color_img = None
        if self.use_img:

            source_color_img = imread_cv2(os.path.join(pair_path, f'camera_0_rgb.png'))
            ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
            # ImgNormTransform = tvf.Compose([ImgNorm])
            Corrupted = tvf.Compose([
                tvf.RandomApply([tvf.ColorJitter(0.5, 0.5, 0.5, hue=0.1)], p=0.5),
                tvf.GaussianBlur(kernel_size=(5, 9), sigma=8), # 0,1,2,4, 8
                tvf.RandomApply([tvf.ToTensor(), tvf2.GaussianNoise(mean=0.0, sigma=0.1), tvf.ToPILImage()], p=0.5),
                tvf.RandomApply([tvf.RandomEqualize()], p=0.3),
                tvf.RandomApply([tvf.RandomSolarize(threshold=192.0)], p=0.2),
                ImgNorm
            ])
            ImgNormTransform = Corrupted
            H, W, C = source_color_img.shape

            # import PIL
            # if self.split == 'train':
            if not isinstance(source_color_img, PIL.Image.Image): 
                source_color_img = PIL.Image.fromarray(source_color_img)
                if self.split == 'train':
                    source_color_img = ColorJitter(source_color_img).numpy()
                else:
                    source_color_img = ImgNormTransform(source_color_img).numpy()
            source_color_img = source_color_img.reshape(C, H, W)  # ONLY FOR SLFM

            # source_color_img, intrinsics = self._crop_resize_if_necessary(
            #     source_color_img,
            #     intrinsics,
            #     resolution,Z
            #     rng=rng
            # )

            target_color_img = imread_cv2(os.path.join(pair_path, f'camera_1_rgb.png'))
            # if self.split == 'train':
            if not isinstance(target_color_img, PIL.Image.Image): 
                target_color_img = PIL.Image.fromarray(target_color_img)
                if self.split == 'train':
                    target_color_img = ColorJitter(target_color_img).numpy()
                else:
                    target_color_img = ImgNormTransform(target_color_img).numpy()
            target_color_img = target_color_img.reshape(C, H, W)  # ONLY FOR SLFM

            embeddings_folder = '-'.join(source_label.split('-')[:2])
            if self.embedding_type == "audio":

                # source_label = f'{"-".join(pair_sub_folders.split("/"))}_0'
                # target_label = f'{"-".join(pair_sub_folders.split("/"))}_1'

                audio_embedding1 = torch.load(os.path.join("./embeddings_slfm_audio/", embeddings_folder,  f"embedding1_{source_label}.pt"))
                audio_embedding2 = torch.load(os.path.join("./embeddings_slfm_audio/", embeddings_folder,  f"embedding2_{target_label}.pt"))
            
            elif self.embedding_type == "doa":
                audio_embedding1 = torch.load(os.path.join("./embeddings_slfm_doa/", embeddings_folder,  f"embedding_{source_label}.pt"))
                audio_embedding2 = torch.load(os.path.join("./embeddings_slfm_doa/", embeddings_folder,  f"embedding_{target_label}.pt"))

            # target_color_img, intrinsics = self._crop_resize_if_necessary(
            #     target_color_img, 
            #     intrinsics, 
            #     resolution, 
            #     rng=rng
            # )

        # relative_angle1to2 = target_rotation - source_rotation
        # relative_angle2to1 = source_rotation - target_rotation
        # if relative_angle1to2 >= 180:
        #     relative_angle1to2 -= 360
        # elif relative_angle1to2 < -180:
        #     relative_angle1to2 += 360
        
        # if relative_angle2to1 >= 180:
        #     relative_angle2to1 -= 360
        # elif relative_angle2to1 < -180:
        #     relative_angle2to1 += 360
        # if not -90 <= relative_angle1to2 <= 90:
        #     raise Exception("PROBLEM", relative_angle1to2, target_rotation, source_rotation, pair_path)

        
        if self.use_audio:
            if self.use_doa:
                if self.use_doa_only_model:
                    views.append(dict(
                        audio_spec=source_spec,
                        label=source_label,
                        doas=source_azimuths.astype(np.float32),
                        instance=os.path.basename(pair_path),
                        sound_size=self.sound_size,
                        save_embedding=True
                    ))
                    views.append(dict(
                        audio_spec=target_spec,
                        label=target_label,
                        doas=target_azimuths.astype(np.float32),
                        instance=os.path.basename(pair_path),
                        sound_size=self.sound_size,
                        save_embedding=True
                    ))
                    return views
                else:
                    views.append(dict(
                        img=source_spec,
                        camera_pose=source_pose.astype(np.float32),
                        dataset="SLfM_HM3D",
                        instance=os.path.basename(pair_path),
                        sound_size=self.sound_size,
                        save_embedding=False,
                        label=source_label,
                        input_doa=source_azimuths.astype(np.float32),
                    # relative_angle=np.array(relative_angle1to2).astype(np.float32),
                    ))
                    views.append(dict( 
                        img=target_spec,
                        camera_pose=target_pose.astype(np.float32),
                        dataset="SLfM_HM3D",
                        instance=os.path.basename(pair_path),
                        sound_size=self.sound_size,
                        save_embedding=False,
                        label=target_label,
                        input_doa=target_azimuths.astype(np.float32),
                        #relative_angle=np.array(relative_angle2to1).astype(np.float32)
                    ))
                    return views
            else:
                views.append(dict(
                    img=source_spec,
                    camera_pose=source_pose.astype(np.float32),
                    dataset="SLfM_HM3D",
                    instance=os.path.basename(pair_path),
                    sound_size=self.sound_size,
                    save_embedding=True,
                    label=source_label,
                # relative_angle=np.array(relative_angle1to2).astype(np.float32),
                ))
                views.append(dict( 
                    img=target_spec,
                    camera_pose=target_pose.astype(np.float32),
                    dataset="SLfM_HM3D",
                    instance=os.path.basename(pair_path),
                    sound_size=self.sound_size,
                    save_embedding=True,
                    label=target_label,
                    #relative_angle=np.array(relative_angle2to1).astype(np.float32)
                ))
                return views
        elif self.use_img:
            if self.embedding_type == "audio" or self.embedding_type == "doa":
                views.append(dict(
                    img=source_color_img.astype(np.float32),
                    camera_pose=source_pose.astype(np.float32),
                    dataset="SLfM_HM3D",
                    instance=os.path.basename(pair_sub_folders),
                    save_embedding=False,
                    audio_embedding=audio_embedding1,
                    label=source_label,
                    img_path = os.path.join(pair_path, f'camera_0_rgb.png'),
                    sound_path = os.path.join(pair_path, "binaural_rirs", "sound_0_camera_0_rir.wav")
                    #relative_angle=np.array(relative_angle1to2).astype(np.float32),
                ))
                views.append(dict( 
                    img=target_color_img.astype(np.float32),
                    camera_pose=target_pose.astype(np.float32),
                    dataset="SLfM_HM3D",
                    instance=os.path.basename(pair_sub_folders),
                    save_embedding=False,
                    audio_embedding=audio_embedding2,
                    label=target_label,
                    img_path = os.path.join(pair_path, f'camera_1_rgb.png'),
                    sound_path = os.path.join(pair_path, "binaural_rirs", "sound_0_camera_1_rir.wav")
                # relative_angle=np.array(relative_angle2to1).astype(np.float32)
                ))
                return views
            else:
                views.append(dict(
                    img=source_color_img.astype(np.float32),
                    camera_pose=source_pose.astype(np.float32),
                    dataset="SLfM_HM3D",
                    instance=os.path.basename(pair_sub_folders),
                    save_embedding=False,
                    label=source_label,
                    img_path = os.path.join(pair_path, f'camera_0_rgb.png'),
                    #relative_angle=np.array(relative_angle1to2).astype(np.float32),
                    # sound_path = os.path.join(pair_path, "binaural_rirs", "sound_0_camera_0_rir.wav")
                ))
                views.append(dict( 
                    img=target_color_img.astype(np.float32),
                    camera_pose=target_pose.astype(np.float32),
                    dataset="SLfM_HM3D",
                    instance=os.path.basename(pair_sub_folders),
                    save_embedding=False,
                    label=target_label,
                    img_path = os.path.join(pair_path, f'camera_1_rgb.png'),
                    # sound_path = os.path.join(pair_path, "binaural_rirs", "sound_0_camera_1_rir.wav")
                # relative_angle=np.array(relative_angle2to1).astype(np.float32)
                ))
            return views
        else:
            raise NotImplementedError("use_audio pr use_img must be True for SLfM_HM3D")

        # #save rendered audio
        # save_folder = os.path.join("slfm_audio", 'render_audios')
        # os.makedirs(save_folder, exist_ok=True)
        # save_path_source = os.path.join(save_folder, f'source_render_audio.wav')
        # save_path_target = os.path.join(save_folder, f'target_render_audio.wav')
        # sf.write(save_path_source, source_render_audio, self.sampling_rate)
        # sf.write(save_path_target, target_render_audio, self.sampling_rate)

        # raise Exception(source_rotation, target_rotation, mono_source_sounds_path, audio_length, mono_source_sound.shape, source_binaural_rir.shape, target_binaural_rir.shape)

