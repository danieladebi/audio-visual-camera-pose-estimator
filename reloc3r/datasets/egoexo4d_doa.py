import numpy as np

import torchvision.transforms as tvf

from reloc3r.datasets.base.easy_dataset import EasyDataset
from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2
import os
import torch

import pickle
import pyroomacoustics as pra
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib

from glob import glob
from math import gcd
from scipy.signal import resample

DATA_ROOT = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/" #'/vision/ikadebi/audio_camera_pose_prediction/frames' 

train_camera_pose_path = "/vision/vision_data_2/egoexo4d_audio/annotations/ego_pose/train/camera_pose"
val_camera_pose_path = "/vision/vision_data_2/egoexo4d_audio/annotations/ego_pose/val/camera_pose" 

def label_to_str(label):
    return '_'.join(label)


mics = {
    "mic0": [0.0340384, -0.0896267, -0.070348],
    "mic1": [-0.0062721, -0.0523816, -0.0304851],
    "mic2": [0.0311063, -0.0167887, -0.0145339],
    "mic3": [-0.0093178, -0.0021818, -0.0068198],
    "mic4": [-0.0070644, -0.098121, -0.0720965],
    "mic5": [-0.0016655, 0.0629508, -0.0809149],
    "mic6": [0.002788, -0.0478431, -0.1658549]
}
L = np.array([
            [mics[mic][0] for mic in mics],
            [mics[mic][1] for mic in mics],
            [mics[mic][2] for mic in mics]
        ])

class EgoExo4D_DOA(BaseStereoViewDataset):

    def __init__(self, n_src=1, sound_size=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT
       # self.split = split
        if self.split is None or self.split == "test" or self.split == "val":
            split = "val"
        else:
            print("currently training", self.split)
            split = "train"

        self.n_src = n_src
        self.sound_size = sound_size
       
        # self.pairs_path = os.path.join("/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data", f"{split}_frame_pairs_list.pickle")
        # with open(self.pairs_path, "rb") as f:
        #     self.frame_pairs_list = pickle.load(f)
    
        # self.doa_file_path = os.path.join("/vision/vision_data_2/egoexo4d_audio/doa_spectra")
        self.data_list = []
        # for file in os.listdir(self.doa_file_path):
        #     print(file)
        #     if file.endswith("azimuths.npy"):
        #         self.data_list.append(file)
        split_path = f"/vision/ikadebi/audio_camera_pose_prediction/splits/{split}_egoexo4d_split.txt"
        with open(split_path, 'r')  as f:
            split = set([line.strip() for line in f.readlines()])

        self.audio_paths = {}
        for root, dirs, files in os.walk(os.path.join(DATA_ROOT, "audio")):

            for vid_name in dirs:
                sound_folder = os.path.join(root, vid_name, "sound")

                # use glob to find all wav files that end with "{self.sound_size}ms.wav"
                pattern = os.path.join(sound_folder, f"*{self.sound_size}ms.wav")
                matched_files = sorted(glob(pattern))

                if vid_name in split and matched_files:
                    if vid_name not in self.audio_paths:
                        self.audio_paths[vid_name] = []
                    self.audio_paths[vid_name].extend(matched_files)
                    # if file.endswith(f"{self.sound_size}ms.wav") and vid_name in split:
                    #     if vid_name not in self.audio_paths:
                    #         self.audio_paths[vid_name] = []
                    #     self.audio_paths[vid_name].append(os.path.join(sound_folder, file))
            # if vid_name not in self.audio_paths and vid_name in split:
            #     self.audio_paths[vid_name] = [os.path.join(root, file) for file in files if file.endswith(f"{self.sound_size}ms.wav")]
       
        for vid_name, paths in self.audio_paths.items():
            for path in paths:
                frame_id = path.split("/")[-1].split("_")[0]
                self.data_list.append(({"vid_name": vid_name, "path": path, "frame_id": frame_id}))

       ## self.data = {}

    def __len__(self):
        return len(self.data_list)

    def normalize_audio(self, samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        samples[samples > 1.] = 1.
        samples[samples < -1.] = -1.
        return samples 

    def _get_views(self, idx, resolution, rng):
        # if idx in self.data and len(self.data) < 100000:
        #     return self.data[idx]
        data = []
        # TODO: add audio spectrogram?
    
        take_name = self.data_list[idx]["vid_name"]
        frame_id = int(self.data_list[idx]["frame_id"])
        sound_path = self.data_list[idx]["path"]

        out_dir = os.path.join(self.data_root, "doa_data", take_name)
        os.makedirs(out_dir, exist_ok=True)
        base = f"{frame_id:06d}_duration_{self.sound_size}ms"
        spec_path = os.path.join(out_dir, f"{base}_spectrogram.npy")
        doa_path = os.path.join(out_dir, f"{base}_azimuths.npy")

        if os.path.exists(spec_path) and os.path.exists(doa_path):
            spectrogram = np.load(spec_path)
            azimuths = np.load(doa_path)

            # matplotlib.use("Agg")

            # az = np.atleast_1d(azimuths).astype(float).flatten()
            # theta = (az % 1.0) * 2 * np.pi
            # r = np.ones_like(theta)

            # fig = plt.figure(figsize=(3, 3), dpi=150)
            # ax = fig.add_subplot(111, projection="polar")
            # ax.scatter(theta, r, c="C1", s=12)
            # ax.set_yticklabels([])
            # ax.set_title(f"{label}", va="bottom", fontsize=8)

            # # Build filename with azimuths (in degrees, 1 decimal) appended
            # deg = (az * 360.0) % 360.0
            # az_str = "-".join(f"{d:.1f}" for d in deg)
            # os.makedirs("visualize_doas", exist_ok=True)
            # plot_path = os.path.join("visualize_doas", f"{label}_azimuths_{az_str}.png")
            # fig.savefig(plot_path, bbox_inches="tight")
            # plt.close(fig)
            # raise Exception("stop")
            data.append(dict(
                dataset="EgoExo4D_DOA",
                audio_spec=spectrogram.astype(np.float32),
                label=f"{take_name}_{frame_id:06d}",
                doas=azimuths.astype(np.float32),
                take_name=take_name,
                sound_size=self.sound_size,
            ))
            data.append(dict(
                dataset="EgoExo4D_DOA",
                audio_spec=spectrogram.astype(np.float32),
                label=f"{take_name}_{frame_id:06d}",
                doas=azimuths.astype(np.float32),
                take_name=take_name,
                sound_size=self.sound_size,
            ))
            return data
        else:
            raise Exception("wth")

        # spectrogram = (spectrogram - spec_min)/(spec_max - spec_min)
        #sound_path = os.path.join(self.data_root, "audio", take_name, "sound", f"{frame_id:06d}_duration_{self.sound_size}ms.wav")
    
        self.data_list[idx]["path"] = self.data_list[idx]["path"]
        eps = 1e-10

        # if not os.path.exists(doa_save_path) or not os.path.exists(spec_save_path):
        waveform, sample_rate = sf.read(sound_path)
        # resample waveform to 16000 using scipy
        if sample_rate != 16000:
            import scipy
            waveform = scipy.signal.resample(waveform, int(waveform.shape[0] / sample_rate * 16000), axis=0)
            waveform = self.normalize_audio(waveform)
            #waveform = waveform.T
            sample_rate = 16000


        segment_length = int(sample_rate * self.sound_size / 1000)
        # if waveform.shape != (segment_length, 7):
        #     data_root_without_audio = self.data_root.replace("/camera_pose_audio_data", "")
        #     full_sound_path = os.path.join(data_root_without_audio, "takes", take_name, "audio")
        # #  full_sound_path = os.path.join(self.data_root, "takes", take_name, "audio")
        #     try:
        #         full_sound_path = glob(os.path.join(full_sound_path, "*.wav"))[0]
        #     except IndexError:
        #         raise FileNotFoundError(f"Full sound file not found for {take_name} at {full_sound_path} at frame_id: {frame_id:06d}")
        #     full_waveform, sample_rate = sf.read(full_sound_path)

        #     start = (sample_rate * (frame_id - 30) ) // 30 - segment_length // 2
        #     end = start + segment_length
        #     waveform = full_waveform[start:end].copy()
        #     del full_waveform
        #     assert waveform.shape == (segment_length, 7), f"Source waveform shape mismatch: {waveform.shape} != {(segment_length, 7)}"

        nfft = 512
        hop = nfft // 8 if self.sound_size == 60 else (nfft // 4 if self.sound_size == 500 else nfft // 4)

        X = pra.transform.stft.analysis(waveform, L=nfft, hop=hop)
        X = X.transpose(2,1,0)
        S = np.abs(X**2)

        # TODO: perfrom normalization

        ref = np.max
        if callable(ref):
            # User supplied a function to calculate reference power
            ref_value = ref(S)
        else:
            ref_value = np.abs(ref)

        spectrogram = 10 * np.log10(np.maximum(S, eps)) - 10 * np.log10(np.maximum(ref_value, eps))
        spectrogram = np.maximum(spectrogram, spectrogram.max() - 80)
        
        doa = pra.doa.normmusic.NormMUSIC(
                    L=L,
                    n_src=self.n_src,
                    fs=sample_rate,
                    nfft=nfft,
                  #  azimuth=np.linspace(0, 2*np.pi, 360, endpoint=False),
                )
        doa.locate_sources(X, num_src=self.n_src)
        azimuths = doa.grid.values.copy()

        

        # save doas to folder
        # np.save(doa_save_path, azimuths)
        # np.save(spec_save_path, spectrogram)
        del S, X, waveform
        # else:
        #     azimuths = np.load(doa_save_path)
        #     spectrogram = np.load(spec_save_path)
        


        spec_max = 0
        spec_min = -80

        spectrogram = (spectrogram - spec_min)/(spec_max - spec_min + eps)
        
        spectrogram = spectrogram.astype(np.float32)

        # C, H, W = spectrogram.shape

        # SpecNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize(0.5, 0.5)])
        # spectrogram = SpecNorm(spectrogram.reshape(H,W,C)).numpy()

        # spectrogram = spectrogram.astype(np.float32).reshape(C,H,W)

        label = f"{take_name}_{frame_id:06d}"
        # audio_embedding1 = torch.load(os.path.join(f"embeddings_100ms/{take_name}/embedding1_{source_label}.pt"))
        # audio_embedding2 = torch.load(os.path.join(f"embeddings_100ms/{take_name}/embedding2_{target_label}.pt"))

       
        azimuths_max = np.max(azimuths)
        azimuths_min = np.min(azimuths)
        azimuths = (azimuths - azimuths_min) / (azimuths_max - azimuths_min )

        # Plot circular graph of azimuths and save with label and azimuths in filename
     

        data.append(dict(
            dataset="EgoExo4D_DOA",
            audio_spec=spectrogram.astype(np.float32),
            label=label,
            doas=azimuths.astype(np.float32),
            take_name=take_name,
            sound_size=self.sound_size,
        ))
        data.append(dict(
            dataset="EgoExo4D_DOA",
            audio_spec=spectrogram.astype(np.float32),
            label=label,
            doas=azimuths.astype(np.float32),
            take_name=take_name,
            sound_size=self.sound_size,
        ))
       
        if not os.path.exists(spec_path):
            np.save(spec_path, spectrogram.astype(np.float32))
        if not os.path.exists(doa_path):
            np.save(doa_path, azimuths.astype(np.float32))

        return data