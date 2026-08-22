import numpy as np

from reloc3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from reloc3r.utils.image import imread_cv2, cv2
import os
import torch
import torchvision.transforms as tvf

from glob import glob
import json

import pyroomacoustics as pra
import soundfile as sf

import pickle
# import tensorflow as tf
# import tensorflow_hub as hub

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

class EgoExo4D(BaseStereoViewDataset):

    def __init__(self, use_slfm_and_doa=False, use_imu=False, use_doa_and_embedding=False, use_direct_doa=False, use_img=False, use_yamnet_embed = False, binaural=False, use_policy_model=False, use_doa=False, use_doa_only_model=False, use_audio=False, embedding_type=None, sound_size=1000, use_slfm=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = DATA_ROOT
        if self.split is None or self.split == "test" or self.split == "val":
            split = "val"
        else:
            print("currently training", self.split)
            split = "train"

        self.embedding_type = embedding_type
        self.use_doa_only_model = use_doa_only_model
        self.use_policy_model = use_policy_model
        self.use_yamnet_embed = use_yamnet_embed
        self.use_direct_doa = use_direct_doa
        self.use_doa_and_embedding = use_doa_and_embedding
        self.use_slfm_and_doa = use_slfm_and_doa
        self.use_img = use_img
        self.use_audio = use_audio or use_policy_model or use_slfm
        self.use_doa = use_doa
        self.sound_size = sound_size
        self.n_src = 1
        self.binaural = binaural
        self.use_slfm = use_slfm
        self.use_imu = use_imu
        # split_path = os.path.join("/vision/ikadebi/audio_camera_pose_prediction/splits", f"{split}_egoexo4d_split.txt")
        # with open(split_path, "r") as f:
        #     self.split_list = f.read().splitlines()

        #self.pairs_path = os.path.join("/vision/vision_data_2/egoexo4d_audio", f"frame_pairs_list_{split}.pickle")
        self.pairs_path = os.path.join("/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data", f"{split}_frame_pairs_list.pickle")
        self.imu_file_path = "imu_data/egoexo4d_imu_data.json" # /cmu_bike01_4/trajectory/open_loop_trajectory.csv
        if self.use_imu:
            self.imu_info = json.load(open(self.imu_file_path, "r"))
            self.imu_torch_dict= {}
            for take_name in self.imu_info.copy():
                self.imu_torch_dict[take_name] = {}
                for frame_id in self.imu_info[take_name]:
                    if self.imu_info[take_name][frame_id] is not None:
                        self.imu_torch_dict[take_name][int(frame_id)] = [float(x) for x in self.imu_info[take_name][frame_id]]
                    else:
                        if str(int(frame_id) - 60) in self.imu_info[take_name] and self.imu_info[take_name][str(int(frame_id) - 60)] is not None:
                            self.imu_torch_dict[take_name][int(frame_id)] = [float(x) for x in self.imu_info[take_name][str(int(frame_id) - 60)]]
                        elif str(int(frame_id) + 60) in self.imu_info[take_name] and self.imu_info[take_name][str(int(frame_id) + 60)] is not None:
                            self.imu_torch_dict[take_name][int(frame_id)] = [float(x) for x in self.imu_info[take_name][str(int(frame_id) + 60)]]
                        else:
                            raise Exception("pain")

        with open(self.pairs_path, "rb") as f:
            self.frame_pairs_list = pickle.load(f)
            ## Get every other frame pair to reduce dataset size
            # new_frame_pairs_list = []
            # k = 0
            # current_source_pose = None
            # current_source_frame_id = None
            # current_take_name = None
            # for pair in self.frame_pairs_list:
            #     if k == 0:
            #         current_source_frame_id = pair["source_frame_id"]
            #         current_source_pose = pair["source_camera_pose"]
            #         current_take_name = pair["take_name"]
            #         k += 1
            #     elif 0 < k < 3:
            #         k += 1
            #         continue
            #     else:
            #         k = 0
            #         if current_take_name != pair["take_name"]:
            #             continue
            #         target_frame_id = pair["target_frame_id"]
            #         target_camera_pose = pair["target_camera_pose"]
            #         new_frame_pairs_list.append({
            #             "source_frame_id": current_source_frame_id,
            #             "target_frame_id": target_frame_id,
            #             "source_camera_pose": current_source_pose,
            #             "target_camera_pose": target_camera_pose,
            #             "take_name": pair["take_name"]
            #         })  
            # self.frame_pairs_list = new_frame_pairs_list

        # TODO: prepare policy model errors and compare it to audio only
        if self.use_policy_model:
            if self.split == "train":                        
                # self.methods_measured = ["vision_only_train", f"audio_only_{self.sound_size}ms_train"] 
                #  self.methods_measured = ["vision_only_train", f"doa_cp_{self.sound_size}ms_train"] 
                self.methods_measured = ["slfm_and_doa_360_train", f"doa_cp_1000ms_train"] 
                # save_dir_train = f"video_prediction_errs_{'_'.join(self.methods_measured)}"
                save_dir_train = "video_prediction_errs_slfm_and_doa_360_train_doa_cp_1000ms_train"
                with open(os.path.join(save_dir_train, "max_errors_per_vid_train.json"), "r") as f:
                    self.max_error_per_vid = json.load(f)
                    self.max_error_per_vid = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in self.max_error_per_vid.items()}
                with open(os.path.join(save_dir_train, "rotation_errors_per_vid_train.json"), "r") as f:
                    self.rotation_error_per_vid = json.load(f)
                    self.rotation_error_per_vid = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in self.rotation_error_per_vid.items()}
                with open(os.path.join(save_dir_train, "translation_errors_per_vid_train.json"), "r") as f:
                    self.translation_error_per_vid = json.load(f)
                    self.translation_error_per_vid = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in self.translation_error_per_vid.items()}

            else:
                #self.methods_measured = ["vision_only", f"audio_only_{self.sound_size}ms"]
                #self.methods_measured = ["vision_only", f"doa_cp_{self.sound_size}ms"]
                self.methods_measured = ["slfm_and_doa_360", f"doa_cp_1000ms"] 

                #save_dir_test = f"video_prediction_errs_{'_'.join(self.methods_measured)}"
                save_dir_test = "video_prediction_errs_slfm_and_doa_360_doa_cp_1000ms"
                with open(os.path.join(save_dir_test, "max_errors_per_vid.json"), "r") as f:
                    self.max_error_per_vid = json.load(f)
                    self.max_error_per_vid = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in self.max_error_per_vid.items()}
                with open(os.path.join(save_dir_test, "rotation_errors_per_vid.json"), "r") as f:
                    self.rotation_error_per_vid = json.load(f)
                    self.rotation_error_per_vid = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in self.rotation_error_per_vid.items()}
                with open(os.path.join(save_dir_test, "translation_errors_per_vid.json"), "r") as f:
                    self.translation_error_per_vid = json.load(f)
                    self.translation_error_per_vid = {k: {eval(kk): vv for kk, vv in v.items()} for k, v in self.translation_error_per_vid.items()}

    def __len__(self):
        return len(self.frame_pairs_list)

    def normalize_audio(self, samples, desired_rms=0.1, eps=1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        samples[samples > 1.] = 1.
        samples[samples < -1.] = -1.
        return samples 

    def _get_views(self, idx, resolution, rng):
        # TODO: add audio spectrogram?
        frame_pair_info = self.frame_pairs_list[idx]
        take_name = frame_pair_info["take_name"]
        source_matrix = np.concatenate((frame_pair_info["source_camera_pose"], np.array([[0,0,0,1]])), axis=0)
        target_matrix = np.concatenate((frame_pair_info["target_camera_pose"], np.array([[0,0,0,1]])), axis=0)
        # source_frame_id, target_frame_id = frame_pair_info["frame_id_pair"]
        source_frame_id = frame_pair_info["source_frame_id"]
        target_frame_id = frame_pair_info["target_frame_id"]

        source_label = f"{take_name}_{source_frame_id:06d}"
        target_label = f"{take_name}_{target_frame_id:06d}"

        # save source and target matrix to this folder:
        # ./result_poses/ground_truths/{take_name}_{source_frame_id}_{target_frame_id}_source_matrix.npy
        # ./result_poses/ground_truths/{take_name}_{source_frame_id}_{target_frame_id}_target_matrix.npy
        os.makedirs(f"./result_poses/ground_truths/{take_name}", exist_ok=True)
        np.save(f"./result_poses/ground_truths/{take_name}/{take_name}_{source_frame_id:06d}.npy", source_matrix)
        np.save(f"./result_poses/ground_truths/{take_name}/{take_name}_{target_frame_id:06d}.npy", target_matrix)

        views = []
        with open(os.path.join(self.data_root, "camera_poses", take_name, "intrinsics.npy"), "rb") as f:
            intrinsics = np.load(f).astype(np.float32)/2

        spec_diff = None
        source_color_img = None
        target_color_img = None
        max_error = None
        if self.use_img:
            source_color_img = imread_cv2(os.path.join(self.data_root, "frames", take_name, f"{source_frame_id:06d}.jpg"))
            source_color_img, intrinsics = self._crop_resize_if_necessary(
                source_color_img, 
                intrinsics, 
                resolution, 
                rng=rng
            )
            target_color_img = imread_cv2(os.path.join(self.data_root, "frames", take_name, f"{target_frame_id:06d}.jpg"))
            target_color_img, intrinsics = self._crop_resize_if_necessary(
                target_color_img, 
                intrinsics, 
                resolution, 
                rng=rng
            )

        source_spectrogram = None
        target_spectrogram = None
        policy_decision = None
        vision_error = 0
        audio_error = 0
        vision_rot_error= 0
        audio_rot_error = 0
        vision_trans_error = 0
        audio_trans_error = 0

        source_yamnet_emb = None
        target_yamnet_emb = None
        
        if self.use_imu:
            source_imu = np.array(self.imu_torch_dict[take_name][source_frame_id]).astype(np.float32)
            target_imu = np.array(self.imu_torch_dict[take_name][target_frame_id]).astype(np.float32)
        if self.use_audio:
            
            source_sound_path = os.path.join(self.data_root, "audio", take_name, "sound", f"{source_frame_id:06d}_duration_{self.sound_size}ms.wav")
            target_sound_path = os.path.join(self.data_root, "audio", take_name, "sound", f"{target_frame_id:06d}_duration_{self.sound_size}ms.wav")

            if not self.use_yamnet_embed:
                source_waveform, sample_rate = sf.read(source_sound_path)
                target_waveform, sample_rate = sf.read(target_sound_path)

                if self.use_slfm:
                    source_waveform = source_waveform[:, -2:]
                    target_waveform = target_waveform[:, -2:]

                    slfm_sr_rate = 16000
                    if sample_rate != slfm_sr_rate:
                        import scipy
                        source_waveform = scipy.signal.resample(source_waveform, int(source_waveform.shape[0] / sample_rate * slfm_sr_rate), axis=0)
                        target_waveform = scipy.signal.resample(target_waveform, int(target_waveform.shape[0] / sample_rate * slfm_sr_rate), axis=0)

                        source_waveform = source_waveform.T
                        target_waveform = target_waveform.T
                        sample_rate = slfm_sr_rate
                else:
                    source_waveform = self.normalize_audio(source_waveform)
                    target_waveform = self.normalize_audio(target_waveform)
                    if self.binaural:
                        source_waveform = source_waveform[:, -2:]
                        target_waveform = target_waveform[:, -2:]

                    segment_length = int(sample_rate * self.sound_size / 1000)
                    if source_waveform.shape[0] != segment_length:
                        data_root_without_audio = self.data_root.replace("/camera_pose_audio_data", "")
                        full_sound_path = os.path.join(data_root_without_audio, "takes", take_name, "audio")
                        full_sound_path = os.path.join(self.data_root, "takes", take_name, "audio")
                        full_sound_path = glob(os.path.join(full_sound_path, "*.wav"))[0]

                        full_source_waveform, sample_rate= sf.read(full_sound_path)

                        start = (sample_rate * source_frame_id ) // 30 - segment_length // 2
                        end = start + segment_length
                        source_waveform = full_source_waveform[start:end].copy()
                        del full_source_waveform
                        assert source_waveform.shape == (segment_length, 7), f"Source waveform shape mismatch: {source_waveform.shape} != {(segment_length, 7)}"

                    if target_waveform.shape[0] != segment_length:
                        data_root_without_audio = self.data_root.replace("/camera_pose_audio_data", "")
                        full_sound_path = os.path.join(data_root_without_audio, "takes", take_name, "audio")
                        full_sound_path = glob(os.path.join(full_sound_path, "*.wav"))[0]

                        target_waveform = source_waveform.copy()
                        target_matrix = source_matrix.copy()

                    nfft = 1023
                    hop = nfft // 4 if self.sound_size == 60 else nfft // 2

                    # TRY DOING BINAURAL AUDIO
                    source_X = pra.transform.stft.analysis(source_waveform, L=nfft, hop=hop) # source_X = pra.transform.stft.analysis(source_waveform[:, -2:], L=nfft, hop=hop)
                    source_X = source_X.transpose(2,1,0)
                    source_S = np.abs(source_X**2)
                    target_X = pra.transform.stft.analysis(target_waveform, L=nfft, hop=hop) # target_X = pra.transform.stft.analysis(target_waveform[:, -2:], L=nfft, hop=hop )
                    target_X = target_X.transpose(2,1,0)
                    target_S = np.abs(target_X**2)

                    ref = np.max
                    if callable(ref):
                        # User supplied a function to calculate reference power
                        source_ref_value = ref(source_S)
                        target_ref_value = ref(target_S)
                    else:
                        source_ref_value = target_ref_value = np.abs(ref)

                    eps = 1e-10
                    source_spectrogram = 10 * np.log10(np.maximum(source_S, eps)) - 10 * np.log10(np.maximum(source_ref_value, eps))
                    target_spectrogram = 10 * np.log10(np.maximum(target_S, eps)) - 10 * np.log10(np.maximum(target_ref_value, eps))

                    source_spectrogram = np.maximum(source_spectrogram, source_spectrogram.max() - 80)
                    target_spectrogram = np.maximum(target_spectrogram, target_spectrogram.max() - 80)

                    del target_S, source_S, target_X, source_X

            if not self.use_slfm:
                # # Pad the 2nd to nearest power of 2
                def pad_spec(spec):
                    shape = spec.shape
                    axis = -1
                    pad_width = [(0, 0)] * spec.ndim

                    if self.sound_size == 1000:
                        time_steps = 96
                    elif self.sound_size == 500:
                        time_steps = 48
                    elif self.sound_size == 60:
                        time_steps = 16
                    else:
                        raise ValueError(f"Unsupported sound size: {self.sound_size}")

                    pad_width[axis] = (0, max(time_steps - shape[axis], 0))
                    if time_steps - shape[axis] < 0:
                        raise Exception(time_steps - shape[axis], target_spectrogram.shape, source_spectrogram.shape, source_waveform.shape, target_waveform.shape,source_frame_id, target_frame_id)
                    spec = np.pad(spec, pad_width, mode='constant', constant_values=-80)
                    return spec

                spec_max = 0
                spec_min = -80

                if self.use_policy_model:
                    # Extract YAMNet embeddings for source and target audio and cache/save them.
                    if self.use_yamnet_embed:
                        try:
                            
                            # Compute embeddings (use cached files if present)
                            yamnet_dir = os.path.join(self.data_root, "yamnet_embeddings", take_name)
                            os.makedirs(yamnet_dir, exist_ok=True)

                            src_emb_path = os.path.join(
                                yamnet_dir, f"{source_frame_id:06d}_duration_{self.sound_size}ms.pt"
                            )
                            tgt_emb_path = os.path.join(
                                yamnet_dir, f"{target_frame_id:06d}_duration_{self.sound_size}ms.pt"
                            )

                            source_yamnet_emb = torch.load(src_emb_path)
                            target_yamnet_emb = torch.load(tgt_emb_path)
                        

                        except Exception as e:
                            source_yamnet_emb = None
                            target_yamnet_emb = None
                            raise e
                    else:
                        SpecNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize(0.5, 0.5)])
                        C, H, W = source_spectrogram.shape
                        source_spectrogram = SpecNorm(source_spectrogram.reshape(H,W,C)).numpy()
                        target_spectrogram = SpecNorm(target_spectrogram.reshape(H,W,C)).numpy()

                        source_spectrogram = source_spectrogram.astype(np.float32).reshape(C,H,W)
                        target_spectrogram = target_spectrogram.astype(np.float32).reshape(C,H,W)

                        spec_diff = np.abs(target_spectrogram - source_spectrogram)
                        # spec_min = -80
                        # spec_max = 0
                        spec_min = spec_diff.min()
                        spec_max = spec_diff.max()
                        
                        spec_diff = (spec_diff - spec_min) / (spec_max - spec_min + 1e-8) * 2 - 1
                        spec_diff = spec_diff.astype(np.float32)

                    vision_error = np.float32(self.max_error_per_vid[take_name][(source_frame_id, target_frame_id)][self.methods_measured[0]])
                    audio_error = np.float32(self.max_error_per_vid[take_name][(source_frame_id, target_frame_id)][self.methods_measured[1]])
                # if self.split != "train":
                    vision_rot_error = np.float32(self.rotation_error_per_vid[take_name][(source_frame_id, target_frame_id)][self.methods_measured[0]])
                    audio_rot_error = np.float32(self.rotation_error_per_vid[take_name][(source_frame_id, target_frame_id)][self.methods_measured[1]])

                    vision_trans_error = np.float32(self.translation_error_per_vid[take_name][(source_frame_id, target_frame_id)][self.methods_measured[0]])
                    audio_trans_error = np.float32(self.translation_error_per_vid[take_name][(source_frame_id, target_frame_id)][self.methods_measured[1]])

                    # NEW STRATEGY
                    DIFF_THRESHOLD = 0 # 10 20
                    ERR_THRESHOLD = 0 # 2.5 5
                    rot_err_diff = audio_rot_error - vision_rot_error
                    trans_err_diff = audio_trans_error - vision_trans_error 

                    if rot_err_diff + trans_err_diff > DIFF_THRESHOLD and audio_rot_error > ERR_THRESHOLD and audio_trans_error > ERR_THRESHOLD:
                        policy_decision = 0  # use vision
                    else:
                        policy_decision = 1  # use audio
                    # END NEW STRATEGY
                    
                    # TODO: edit this condition
                    # if vision_error < audio_error :
                    #     policy_decision = 0  # use vision
                    # else:
                    #     policy_decision = 1
                    
                

                # source_spectrogram = (source_spectrogram - spec_min)/(spec_max - spec_min)
                # target_spectrogram = (target_spectrogram - spec_min)/(spec_max - spec_min)
                else:
                    source_spectrogram = pad_spec(source_spectrogram)
                    target_spectrogram = pad_spec(target_spectrogram)

                    C, H, W = source_spectrogram.shape

                    SpecNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize(0.5, 0.5)])
                    source_spectrogram = SpecNorm(source_spectrogram.reshape(H,W,C)).numpy()
                    target_spectrogram = SpecNorm(target_spectrogram.reshape(H,W,C)).numpy()

                    source_spectrogram = source_spectrogram.astype(np.float32).reshape(C,H,W)
                    target_spectrogram = target_spectrogram.astype(np.float32).reshape(C,H,W)
            
        source_doa_save_path = os.path.join(self.data_root, "doa", take_name, f"doa_{source_frame_id:06d}_duration_{self.sound_size}ms.npy")
        target_doa_save_path = os.path.join(self.data_root, "doa", take_name, f"doa_{target_frame_id:06d}_duration_{self.sound_size}ms.npy")
        if self.use_doa_only_model:
            if not (os.path.exists(source_doa_save_path) and os.path.exists(target_doa_save_path)):
                os.makedirs(os.path.dirname(source_doa_save_path), exist_ok=True)
                os.makedirs(os.path.dirname(target_doa_save_path), exist_ok=True)
                source_sound_path = os.path.join(self.data_root, "audio", take_name, "sound", f"{source_frame_id:06d}_duration_{self.sound_size}ms.wav")
                target_sound_path = os.path.join(self.data_root, "audio", take_name, "sound", f"{target_frame_id:06d}_duration_{self.sound_size}ms.wav")

                source_waveform, sample_rate = sf.read(source_sound_path)
                target_waveform, sample_rate = sf.read(target_sound_path)

                segment_length = int(sample_rate * self.sound_size / 1000)
                if source_waveform.shape != (segment_length, 7):
                    data_root_without_audio = self.data_root.replace("/camera_pose_audio_data", "")
                    full_sound_path = os.path.join(data_root_without_audio, "takes", take_name, "audio")
                #  full_sound_path = os.path.join(self.data_root, "takes", take_name, "audio")
                    try:
                        full_sound_path = glob(os.path.join(full_sound_path, "*.wav"))[0]
                    except IndexError:
                        raise FileNotFoundError(f"Full sound file not found for {take_name} at {full_sound_path} at frame_id: {source_frame_id:06d}")
                    full_waveform, sample_rate = sf.read(full_sound_path)

                    start = (sample_rate * (source_frame_id - 30) ) // 30 - segment_length // 2
                    end = start + segment_length
                    source_waveform = full_waveform[start:end].copy()
                    #  del full_waveform
                    assert source_waveform.shape == (segment_length, 7), f"Source waveform shape mismatch: {source_waveform.shape} != {(segment_length, 7)}"
                if target_waveform.shape != (segment_length, 7):
                    data_root_without_audio = self.data_root.replace("/camera_pose_audio_data", "")
                    full_sound_path = os.path.join(data_root_without_audio, "takes", take_name, "audio")

                    try:
                        full_sound_path = glob(os.path.join(full_sound_path, "*.wav"))[0]
                    except IndexError:
                        raise FileNotFoundError(f"Full sound file not found for {take_name} at {full_sound_path} at frame_id: {target_frame_id:06d}")
                    full_waveform, sample_rate = sf.read(full_sound_path)   
                    start = (sample_rate * (target_frame_id - 30) ) // 30 - segment_length // 2
                    end = start + segment_length
                    target_waveform = full_waveform[start:end].copy()
                #    del full_waveform
                    assert target_waveform.shape == (segment_length, 7), f"Target waveform shape mismatch: {target_waveform.shape} != {(segment_length, 7)}"

                if not self.use_yamnet_embed:
                    nfft = 1024
                    hop = nfft // 8 if self.sound_size == 60 else (nfft // 4 if self.sound_size == 500 else nfft // 2)

                    source_X = pra.transform.stft.analysis(source_waveform, L=nfft, hop=hop)
                    source_X = source_X.transpose(2,1,0)
                    source_S = np.abs(source_X**2)

                    target_X = pra.transform.stft.analysis(target_waveform, L=nfft, hop=hop )
                    target_X = target_X.transpose(2,1,0)
                    target_S = np.abs(target_X**2)

                    # TODO: perform normalization
                    source_doa = pra.doa.normmusic.NormMUSIC(
                                L=L,
                                n_src=self.n_src,
                                fs=sample_rate,
                                nfft=nfft,
                            )
                    source_doa.locate_sources(source_X, num_src=self.n_src)
                    source_azimuths = source_doa.grid.values.copy()
                    if source_azimuths.shape[0] != 360:
                        raise ValueError(f"Source DOA shape mismatch: {source_azimuths.shape} != (360,)", "take_name:", take_name, "source_frame_id:", source_frame_id, source_X.shape)

                    target_doa = pra.doa.normmusic.NormMUSIC(
                                L=L,
                                n_src=self.n_src,
                                fs=sample_rate,
                                nfft=nfft
                            )
                    target_doa.locate_sources(target_X, num_src=self.n_src)
                    target_azimuths = target_doa.grid.values.copy()
                    if target_azimuths.shape[0] != 360:
                        raise ValueError(f"Target DOA shape mismatch: {target_azimuths.shape} != (360,)", "take_name:", take_name, "target_frame_id:", target_frame_id, target_X.shape)

                        # save doas to folder
                    if not (os.path.exists(source_doa_save_path) and os.path.exists(target_doa_save_path)):
                        np.save(source_doa_save_path, source_azimuths)
                        np.save(target_doa_save_path, target_azimuths)

                # del source_S, source_X, target_S, target_X, source_waveform, target_waveform
            else:
                source_azimuths = np.load(source_doa_save_path)
                target_azimuths = np.load(target_doa_save_path)

                if source_azimuths.shape[0] != 360:
                    raise ValueError(f"Source DOA shape mismatch: {source_azimuths.shape} != (360,)")
                if target_azimuths.shape[0] != 360:
                    raise ValueError(f"Target DOA shape mismatch: {target_azimuths.shape} != (360,)")


        # audio_embedding1 = torch.load(os.path.join(f"embeddings_100ms/{take_name}/embedding1_{source_label}.pt"))
        if self.embedding_type is not None:
            if "doa" in self.embedding_type.lower():
                
                if self.use_slfm_and_doa:
                    audio_embedding1 = torch.load(os.path.join(f"embeddings_slfm_egoexo4d/{take_name}/embedding1_{source_label}.pt")).astype(np.float32).squeeze()
                    audio_embedding2 = torch.load(os.path.join(f"embeddings_slfm_egoexo4d/{take_name}/embedding2_{target_label}.pt")).astype(np.float32).squeeze()
                    
                    doa1 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/doa_{source_label}.pt")).astype(np.float32).squeeze()
                    doa2 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/doa_{target_label}.pt")).astype(np.float32).squeeze()

                    # def _zscore(x, eps=1e-8):
                    #     x = np.asarray(x, dtype=np.float32)
                    #     mean = float(x.mean())
                    #     std = float(x.std())
                    #     if std < eps:
                    #         return np.zeros_like(x, dtype=np.float32)
                    #     return (x - mean) / (std + eps)

                    
                    # audio_embedding1 = _zscore(audio_embedding1)
                    # audio_embedding2 = _zscore(audio_embedding2)
                    # doa1 = _zscore(doa1)            
                    # doa2 = _zscore(doa2)


                    audio_embedding1 = np.concatenate((audio_embedding1, doa1), axis=-1)
                    audio_embedding2 = np.concatenate((audio_embedding2, doa2), axis=-1)

                elif self.use_doa_and_embedding:
                    audio_embedding1 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/embedding_{source_label}.pt")).astype(np.float32).squeeze()
                    audio_embedding2 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/embedding_{target_label}.pt")).astype(np.float32).squeeze()
                    doa1 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/doa_{source_label}.pt")).astype(np.float32).squeeze()
                    doa2 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/doa_{target_label}.pt")).astype(np.float32).squeeze()

                    # Normalize audio_embedding and DOA separately before concatenation
                    def _zscore(x, eps=1e-8):
                        x = np.asaRrray(x, dtype=np.float32)
                        mean = float(x.mean())
                        std = float(x.std())
                        if std < eps:
                            return np.zeros_like(x, dtype=np.float32)
                        return (x - mean) / (std + eps)

                    audio_embedding1 = _zscore(audio_embedding1)
                    doa1 = _zscore(doa1)
                    audio_embedding2 = _zscore(audio_embedding2)
                    doa2 = _zscore(doa2)

                    audio_embedding1 = np.concatenate((audio_embedding1, doa1), axis=0)
                    audio_embedding2 = np.concatenate((audio_embedding2, doa2), axis=0)
                elif self.use_direct_doa:
                    audio_embedding1 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/doa_{source_label}.pt")).astype(np.float32).squeeze()
                    audio_embedding2 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/doa_{target_label}.pt")).astype(np.float32).squeeze()
                else:
                    audio_embedding1 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/embedding_{source_label}.pt")).astype(np.float32)
                    audio_embedding2 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/embedding_{target_label}.pt")).astype(np.float32)
            elif "slfm" in self.embedding_type.lower():
                audio_embedding1 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/embedding1_{source_label}.pt")).astype(np.float32)
                audio_embedding2 = torch.load(os.path.join(f"{self.embedding_type}/{take_name}/embedding2_{target_label}.pt")).astype(np.float32)
            else:
                audio_embedding1 = torch.load(os.path.join(f"embeddings_{self.sound_size}ms/{take_name}/embedding1_{source_label}.pt")).astype(np.float32)
                audio_embedding2 = torch.load(os.path.join(f"embeddings_{self.sound_size}ms/{take_name}/embedding2_{target_label}.pt")).astype(np.float32)
        else:
            if self.use_doa:
                audio_embedding1 = torch.load(os.path.join(f"embeddings_doa_{self.sound_size}ms/{take_name}/embedding_{source_label}.pt")).astype(np.float32)
                audio_embedding2 = torch.load(os.path.join(f"embeddings_doa_{self.sound_size}ms/{take_name}/embedding_{target_label}.pt")).astype(np.float32)
            else:
                audio_embedding1 = None
                audio_embedding2 = None

        if self.use_img or self.use_audio:
            if self.use_imu:
                views.append(dict(
                    img=source_color_img if self.use_img else source_spectrogram,
                    camera_intrinsics=intrinsics,
                    camera_pose = source_matrix.astype(np.float32),
                    dataset="EgoExo4D",
                    label=source_label,
                    instance=f"{source_frame_id:06d}.jpg",
                    take_name=take_name,
                    source_frame_id =source_frame_id,
                    save_embedding=False,
                    imu=source_imu,
                ))
                views.append(dict(
                    img=target_color_img if self.use_img else target_spectrogram,
                    camera_intrinsics=intrinsics,
                    camera_pose = target_matrix.astype(np.float32),
                    dataset="EgoExo4D",
                    label=target_label,
                    instance=f"{target_frame_id:06d}.jpg",
                    take_name=take_name,
                    target_frame_id =target_frame_id,
                    save_embedding=False,
                    imu=target_imu,
                ))
                return views
            if self.use_slfm:
                source_view = dict(
                    img=source_color_img,
                    waveform=source_waveform.astype(np.float32), 
                    camera_intrinsics=intrinsics,
                    camera_pose = source_matrix.astype(np.float32),
                    dataset="EgoExo4D",
                    label=source_label,
                    instance=f"{source_frame_id:06d}.jpg",
                    take_name=take_name,
                    source_frame_id =source_frame_id,
                    sound_size=self.sound_size,
                    save_embedding=False,
                    #audio_embedding=audio_embedding1,
                )
                target_view = dict(
                    img=target_color_img,
                    waveform=target_waveform.astype(np.float32), 
                    camera_intrinsics=intrinsics,
                    camera_pose = target_matrix.astype(np.float32),
                    dataset="EgoExo4D",
                    label=target_label,
                    instance=f"{target_frame_id:06d}.jpg",
                    take_name=take_name,
                    target_frame_id =target_frame_id,
                    sound_size=self.sound_size,
                    save_embedding=False,
                    #audio_embedding=audio_embedding2,
                )
                views.append(source_view)
                views.append(target_view)
                return views
            if self.use_policy_model:
                views.append(dict(
                    img=spec_diff,
                    policy_decision=policy_decision,
                    camera_intrinsics=intrinsics,
                    dataset="EgoExo4D",
                    label=source_label,
                    instance=f"{source_frame_id:06d}.jpg",
                    take_name=take_name,
                    source_frame_id =source_frame_id,
                    sound_size=self.sound_size,
                    save_embedding=False,
                    vision_error=vision_error,
                    audio_error=audio_error,
                    vision_rot_error=vision_rot_error,
                    audio_rot_error=audio_rot_error,
                    vision_trans_error=vision_trans_error,
                    audio_trans_error=audio_trans_error,
                 #   yamnet_embedding=source_yamnet_emb,
                ))
                views.append(dict(
                    img=spec_diff,
                    policy_decision=policy_decision,
                    camera_intrinsics=intrinsics,
                    dataset="EgoExo4D",
                    label=target_label,
                    instance=f"{target_frame_id:06d}.jpg",
                    take_name=take_name,
                    target_frame_id =target_frame_id,
                    sound_size=self.sound_size,
                    save_embedding=False,
                    vision_error=vision_error,
                    audio_error=audio_error,
                    vision_rot_error=vision_rot_error,
                    audio_rot_error=audio_rot_error,
                    vision_trans_error=vision_trans_error,
                    audio_trans_error=audio_trans_error,
                #    yamnet_embedding=target_yamnet_emb,
                ))

                return views
            if self.embedding_type is None:
                views.append(dict(
                    img=source_color_img if self.use_img else source_spectrogram,
                    # img=source_spectrogram,
                    camera_intrinsics=intrinsics,
                    camera_pose = source_matrix.astype(np.float32),
                    dataset="EgoExo4D",
                    label=source_label,
                    instance=f"{source_frame_id:06d}.jpg",
                    take_name=take_name,
                    source_frame_id =source_frame_id,
                    save_embedding=False,
                ))
                views.append(dict(
                    img=target_color_img if self.use_img else target_spectrogram,
                #  img=target_spectrogram,
                    camera_intrinsics=intrinsics,
                    camera_pose = target_matrix.astype(np.float32),
                    dataset="EgoExo4D",
                    label=target_label,
                    instance=f"{target_frame_id:06d}.jpg",
                    take_name=take_name,
                    target_frame_id =target_frame_id,
                    save_embedding=False,
                # audio_spec=target_spectrogram,
                ))
                return views
            views.append(dict(
                img=source_color_img if self.use_img else source_spectrogram,
            # img=source_spectrogram,
                camera_intrinsics=intrinsics,
                camera_pose = source_matrix.astype(np.float32),
                dataset="EgoExo4D",
                label=source_label,
                instance=f"{source_frame_id:06d}.jpg",
                take_name=take_name,
                source_frame_id =source_frame_id,
                sound_size=self.sound_size,
                save_embedding=False,
            # audio_spec=source_spectrogram,
                audio_embedding=audio_embedding1,
            ))
            if self.use_imu:
                views[-1]["imu"] = source_imu

            views.append(dict(
                img=target_color_img if self.use_img else target_spectrogram,
            #  img=target_spectrogram,
                camera_intrinsics=intrinsics,
                camera_pose = target_matrix.astype(np.float32),
                dataset="EgoExo4D",
                label=target_label,
                instance=f"{target_frame_id:06d}.jpg",
                take_name=take_name,
                target_frame_id =target_frame_id,
                sound_size=self.sound_size,
                save_embedding=False,
            # audio_spec=target_spectrogram,
                audio_embedding=audio_embedding2,
            ))
            if self.use_imu:
                views[-1]["imu"] = target_imu

            if audio_embedding1 is None or audio_embedding2 is None:
                del views[0]["audio_embedding"]
                del views[1]["audio_embedding"]
        else:
            if self.use_doa_only_model:
                views.append(dict(
                    camera_pose = source_matrix.astype(np.float32),
                    dataset="EgoExo4D",
                    label=source_label,
                    instance=f"{source_frame_id:06d}.jpg",
                    take_name=take_name,
                    source_frame_id =source_frame_id,
                    sound_size=self.sound_size,
                    save_embedding=False,
                # audio_spec=source_spectrogram,
                    input_doa=source_azimuths.astype(np.float32),
                ))
                views.append(dict(
                    camera_pose = target_matrix,
                    dataset="EgoExo4D",
                    label=target_label,
                    instance=f"{target_frame_id:06d}.jpg",
                    take_name=take_name,
                    target_frame_id =target_frame_id,
                    sound_size=self.sound_size,
                    save_embedding=False,
                # audio_spec=target_spectrogram,
                    input_doa=target_azimuths.astype(np.float32),
                ))
            else:
                views.append(dict(
                    camera_pose = source_matrix.astype(np.float32),
                    dataset="EgoExo4D",
                    label=source_label,
                    instance=f"{source_frame_id:06d}.jpg",
                    take_name=take_name,
                    source_frame_id =source_frame_id,
                    sound_size=self.sound_size,
                    save_embedding=False,
                # audio_spec=source_spectrogram,
                    audio_embedding=audio_embedding1,
                ))

                views.append(dict(
                    camera_pose = target_matrix.astype(np.float32),
                    dataset="EgoExo4D",
                    label=target_label,
                    instance=f"{target_frame_id:06d}.jpg",
                    take_name=take_name,
                    target_frame_id =target_frame_id,
                    sound_size=self.sound_size,
                    save_embedding=False,
                # audio_spec=target_spectrogram,
                    audio_embedding=audio_embedding2,
                ))

        return views
