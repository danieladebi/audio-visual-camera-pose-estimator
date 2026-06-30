import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

from reloc3r.utils.image import imread_cv2, cv2
from tqdm import tqdm
import pickle 
import json
import soundfile as sf
import librosa

import matplotlib.pyplot as plt
from librosa.sequence import dtw


def show_pair(info, title_prefix):
    src = imread_cv2(info["source_frame_path"])
    tgt = imread_cv2(info["target_frame_path"])
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,5))
    plt.suptitle(f"{title_prefix} SSIM={info['similarity']:.4f}")
    plt.subplot(1,2,1); plt.imshow(src); plt.axis("off"); plt.title("source")
    plt.subplot(1,2,2); plt.imshow(tgt); plt.axis("off"); plt.title("target")
    os.makedirs("ssim_images", exist_ok=True)
    plt.savefig(f"ssim_images/{title_prefix}_SSIM_{info['similarity']:.4f}.png")
    plt.close()


DATA_ROOT = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/" 
frame_path = os.path.join(DATA_ROOT, "frames")
audio_path = os.path.join(DATA_ROOT, "audio") # take_name, "sound", f"{source_frame_id:06d}_duration_1000ms.wav

#train_pairs_list = os.path.join(DATA_ROOT, f"train_frame_pairs_list.pickle")
val_pairs_list_path = os.path.join(DATA_ROOT, f"val_frame_pairs_list.pickle")
val_pairs_list = pickle.load(open(val_pairs_list_path, "rb"))

# TODO: get audio info to filter through

sim_list = []
max_ssim_info = {}
max_ssim_value = -np.inf
min_ssim_value = np.inf
min_ssim_info = {}

ssim_info_dict = {}

intensity_diffs = []
intensity_diff_max = -np.inf
intensity_diff_min = np.inf

intensity_diff_max_info = {}
intensity_diff_min_info = {}

lower_intensity_diff_list = []
higher_intensity_diff_list = []

for frame_pair_info in tqdm(val_pairs_list):
    take_name = frame_pair_info['take_name']
    source_frame_id = frame_pair_info['source_frame_id']
    target_frame_id = frame_pair_info['target_frame_id']
    
    source_frame_path = os.path.join(frame_path,  take_name, f"{source_frame_id:06d}.jpg")
    target_frame_path = os.path.join(frame_path,  take_name, f"{target_frame_id:06d}.jpg")
    source_frame = imread_cv2(source_frame_path)
    target_frame = imread_cv2(target_frame_path)
    
    g1 = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    
    similarity = ssim(g1, g2, data_range=255)
    sim_list.append(similarity)
    
    if similarity > 0.75 or similarity < 0.4:
        #print(f"Take: {take_name}, Source ID: {source_frame_id}, Target ID: {target_frame_id}, SSIM: {similarity:.4f}")
        rand = np.random.rand()
        # if rand < 0.05:  # only show 5% of the high and low similarity pairs
        #     show_pair({
        #         "take_name": take_name,
        #         "source_frame_path": source_frame_path,
        #         "target_frame_path": target_frame_path,
        #         "similarity": similarity
        #     }, title_prefix=f"High_{take_name}_{source_frame_id}_{target_frame_id}" if similarity > 0.75 else f"Low_{take_name}_{source_frame_id}_{target_frame_id}")
        
    if similarity > max_ssim_value:
        max_ssim_value = similarity
        max_ssim_info = {
            "take_name": take_name,
            "source_frame_id": source_frame_id,
            "target_frame_id": target_frame_id,
            "similarity": similarity,
            "source_frame_path": source_frame_path,
            "target_frame_path": target_frame_path
        }
    if similarity < min_ssim_value:
        min_ssim_value = similarity
        min_ssim_info = {
            "take_name": take_name,
            "source_frame_id": source_frame_id,
            "target_frame_id": target_frame_id,
            "similarity": similarity,
            "source_frame_path": source_frame_path,
            "target_frame_path": target_frame_path
        }
    
    ssim_info = {
        "take_name": take_name,
        "source_frame_id": int(source_frame_id),
        "target_frame_id": int(target_frame_id),
        "similarity": float(similarity),
    }
    ssim_info_dict[str((take_name, source_frame_id, target_frame_id))] = ssim_info
    
    
    source_sound_path = os.path.join(audio_path, take_name, "sound", f"{source_frame_id:06d}_duration_1000ms.wav")
    target_sound_path = os.path.join(audio_path, take_name, "sound", f"{target_frame_id:06d}_duration_1000ms.wav")

    def _load_wav(path):
        if not os.path.exists(path) or sf is None:
            return None, None
        y, sr = sf.read(path)
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        return y, sr

    src_audio, src_sr = _load_wav(source_sound_path)
    tgt_audio, tgt_sr = _load_wav(target_sound_path)
    
 
    ## TODO: compute MFCC similarity

    src_mfcc = librosa.feature.mfcc(y=src_audio, sr=src_sr, n_mfcc=20)
    tgt_mfcc = librosa.feature.mfcc(y=tgt_audio, sr=tgt_sr, n_mfcc=20)
    
    # Compute DTW
    D, wp = dtw(src_mfcc, tgt_mfcc, subseq=True)
    mfcc_sim = D[-1, -1]  # DTW distance

    key = str((take_name, source_frame_id, target_frame_id))
    # ssim_info_dict[key]["source_audio_log_mel"] =  float(src_log_mel)
    # ssim_info_dict[key]["target_audio_log_mel"] = None if tgt_log_mel is None else float(tgt_log_mel)
    mfcc_sim_min = 1638.0897458262125
    mfcc_sim_max = 29996.055020222866
    normalized_mfcc_sim = (mfcc_sim - mfcc_sim_min) / (mfcc_sim_max - mfcc_sim_min)
    ssim_info_dict[key]["log_mel_intensity_difference"] = normalized_mfcc_sim
    intensity_diffs.append(normalized_mfcc_sim)
    intensity_diff_max_info["take_name"] = take_name
    intensity_diff_max_info["source_frame_id"] = source_frame_id
    intensity_diff_max_info["target_frame_id"] = target_frame_id
    intensity_diff_max_info["intensity_difference"] = normalized_mfcc_sim
    if normalized_mfcc_sim > 0.95:
        intensity_diff_max = normalized_mfcc_sim
        #  higher_intensity_diff_list.append((take_name, source_frame_id, target_frame_id, normalized_mfcc_sim))
        intensity_diff_max_info = {
            "take_name": take_name,
            "source_frame_id": source_frame_id,
            "target_frame_id": target_frame_id,
            "intensity_difference": normalized_mfcc_sim
        }
    if normalized_mfcc_sim < 0.05:
        intensity_diff_min = normalized_mfcc_sim
        # lower_intensity_diff_list.append((take_name, source_frame_id, target_frame_id, normalized_mfcc_sim))
        intensity_diff_min_info = {
            "take_name": take_name,
            "source_frame_id": source_frame_id,
            "target_frame_id": target_frame_id,
            "intensity_difference": normalized_mfcc_sim
        }
    
print("Average SSIM:", np.mean(sim_list), "Std SSIM:", np.std(sim_list))
print("Min SSIM:", np.min(sim_list))
print("25th Percentile SSIM:", np.percentile(sim_list, 25))
print("50th Percentile SSIM:", np.percentile(sim_list, 50))
print("75th Percentile SSIM:", np.percentile(sim_list, 75))
print("Max SSIM:", np.max(sim_list))

print("Min SSIM Info:", min_ssim_info)
print("Max SSIM Info:", max_ssim_info)


print("Average Intensity Difference:", np.mean(intensity_diffs))
print("Min Intensity Difference:", np.min(intensity_diffs))
print("25th Percentile Intensity Difference:", np.percentile(intensity_diffs, 25))
print("50th Percentile Intensity Difference:", np.percentile(intensity_diffs, 50))
print("75th Percentile Intensity Difference:", np.percentile(intensity_diffs, 75))
print("Max Intensity Difference:", np.max(intensity_diffs))


print("min Intensity Difference Info:", intensity_diff_min_info)
print("max Intensity Difference Info:", intensity_diff_max_info)

val_ssim_info_path = os.path.join("ssim_info", "val_frame_pairs_ssim_info.json")
os.makedirs(os.path.dirname(val_ssim_info_path), exist_ok=True)
with open(val_ssim_info_path, "w") as f:
    json.dump(ssim_info_dict, f, indent=4)

# TODO: display images that fit min and max criteria
# import matplotlib.pyplot as plt

# def show_pair(info, title_prefix):
#     src = imread_cv2(info["source_frame_path"])
#     tgt = imread_cv2(info["target_frame_path"])
#     src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
#     tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
#     plt.figure(figsize=(10,5))
#     plt.suptitle(f"{title_prefix} SSIM={info['similarity']:.4f}")
#     plt.subplot(1,2,1); plt.imshow(src); plt.axis("off"); plt.title("source")
#     plt.subplot(1,2,2); plt.imshow(tgt); plt.axis("off"); plt.title("target")
#     plt.show()

# show_pair(min_ssim_info, "Min")
# show_pair(max_ssim_info, "Max")
