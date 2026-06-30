import numpy as np

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2
from reloc3r.utils.image import ImgNorm
import os
import torch
import torchvision.transforms as tvf
import scipy
from scipy.signal import fftconvolve

from glob import glob
import matplotlib

import pyroomacoustics as pra
import soundfile as sf
import json

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

class SLfM_HM3D_DOA(BaseStereoViewDataset):
    
    def __init__(self, sound_size=2550, n_src=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT

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
                source_binaural_rir_path = os.path.join(pair_path, "camera_0_rgb.png")
                target_binaural_rir_path = os.path.join(pair_path, "camera_1_rgb.png")
                updated_list.append(source_binaural_rir_path)
                updated_list.append(target_binaural_rir_path)
        self.data_list = updated_list

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
        return len(self.data_list)

    def _get_views(self, idx, resolution, rng):
        views = []
        pair_path = self.data_list[idx]
        pair_sub_folders = pair_path.split("rotation/")[-1]
        pair_path = os.path.join(self.data_root, pair_sub_folders)
        pair_path, idx_name = pair_path.split("camera")
        metadata_path = os.path.join(pair_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        idx = 0 if "0" in idx_name else 1 
        label = f'{"-".join(pair_sub_folders.split("/")[:-1])}_{idx}'

        spec = None

        binaural_rir_path = os.path.join(pair_path, "binaural_rirs", f"sound_0_camera_{idx}_rir.wav")

        binaural_rir, _ = sf.read(binaural_rir_path, dtype='float32', always_2d=True)

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
        binaural_convolved = np.array([fftconvolve(mono_source_sound, binaural_rir[:, channel]) for channel in range(binaural_rir.shape[-1])])
        binaural_convolved = binaural_convolved[:, :audio_length]
        render_audio = binaural_convolved

        del mono_source_sound

        # compute spectrogram
        C, L = render_audio.shape

        frames = 256
        hop_length = int(L // (frames - 1))

        n_fft = 512
        spec = torch.stft(
            input=torch.tensor(render_audio),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=400,
            return_complex=True
        )

        doa = pra.doa.normmusic.NormMUSIC(
                    L=locs,
                    n_src=self.n_src,
                    fs=self.sampling_rate,
                    nfft=n_fft,
                    azimuth=np.linspace(-90.,90.,180)*np.pi/180
                )
        # doa = pra.doa.srp.SRP(
        #             L=locs,
        #             n_src=self.n_src,
        #             fs=self.sampling_rate,
        #             nfft=n_fft,
        #         )
        X = pra.transform.stft.analysis(render_audio.T, L=n_fft, hop=hop_length)
        X = X.transpose(2, 1, 0)  # shape: (n_mics, n_frames, n_freq_bins)

        doa.locate_sources(X, num_src=1)
        azimuths = doa.grid.values.copy()
        
        azimuths = np.array(azimuths).astype(np.float32)  # in degrees

        azimuths = (azimuths - azimuths.min()) / (azimuths.max() - azimuths.min())

        del render_audio, binaural_convolved, binaural_rir

        spec = spec.contiguous() #.view(C, *source_spec.shape[1:])

        spec = torch.abs(spec)

        spec = self.normalize_magnitude(spec)[:, :, :]

        spec = spec[:, :-1, :]

        views.append(dict(
            audio_spec=spec,
            label=label,
            doas=azimuths.astype(np.float32),
            instance=os.path.basename(pair_path),
            sound_size=self.sound_size,
            save_embedding=True,
            dataset="SLFM_HM3D_DOA"
        ))
        views.append(dict(
            audio_spec=spec,
            label=label,
            doas=azimuths.astype(np.float32),
            instance=os.path.basename(pair_path),
            sound_size=self.sound_size,
            save_embedding=True,
            dataset="SLFM_HM3D_DOA"
        ))
        return views
