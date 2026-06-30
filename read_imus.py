import json
import csv
import os
import numpy as np
import pickle

from tqdm import tqdm
from pprint import pprint

train_pairs_list_path = os.path.join("/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data", f"train_frame_pairs_list.pickle")
val_pairs_list_path = os.path.join("/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data", f"val_frame_pairs_list.pickle")

# train_pairs_path = "/vision/vision_data_2/egoexo4d_audio/frame_pairs_train.pickle"
# val_pairs_path = "/vision/vision_data_2/egoexo4d_audio/frame_pairs_val.pickle"

imu_file_path = "/vision/vision_data_2/EgoExo4D_public_v1/takes/" # /cmu_bike01_4/trajectory/open_loop_trajectory.csv

train_pairs_list = pickle.load(open(train_pairs_list_path, "rb"))
val_pairs_list = pickle.load(open(val_pairs_list_path, "rb"))

# train_pairs = pickle.load(open(train_pairs_path, "rb"))
# val_pairs = pickle.load(open(val_pairs_path, "rb"))
train_pairs = {}
val_pairs = {}
for pair in train_pairs_list:
    take_name = pair['take_name']
    frame_id_pair = (pair['source_frame_id'], pair['target_frame_id'])
    if take_name not in train_pairs:
        train_pairs[take_name] = []
    train_pairs[take_name].append({
        "frame_id_pair": frame_id_pair
    })

for pair in val_pairs_list:
    take_name = pair['take_name']
    frame_id_pair = (pair['source_frame_id'], pair['target_frame_id'])
    if take_name not in val_pairs:
        val_pairs[take_name] = []
    val_pairs[take_name].append({
        "frame_id_pair": frame_id_pair
    })

# print(len(val_pairs))

# print(val_pairs)

# with open(imu_csv_path, 'r') as csvfile:
#     csvreader = csv.DictReader(csvfile)
#     for i, row in enumerate(csvreader):
#         if i % 1000 != 500:
#             continue

#         filtered_row = {
#             k: v for k, v in row.items() 
#             if k not in ["tracking_timestamp_us", "utc_timestamp_ns", "session_uid", "quality_score"] and not k.startswith("gravity_")
#         }
#         imu_data.append(filtered_row)
#         pprint(filtered_row)
#         print(len(imu_data), len(filtered_row))

imu_data_dict = {}
imu_output_dict = {}
for frame_pair_info in tqdm(train_pairs_list + val_pairs_list):
    take_name = frame_pair_info['take_name']
    source_frame_id = frame_pair_info['source_frame_id']
    target_frame_id = frame_pair_info['target_frame_id']
    if take_name in train_pairs:
        frame_list = sorted(list(set([pair["frame_id_pair"][0] for pair in train_pairs[take_name]] + [pair["frame_id_pair"][1] for pair in train_pairs[take_name]])))
    elif take_name in val_pairs:
        frame_list = sorted(list(set([pair["frame_id_pair"][0] for pair in val_pairs[take_name]] + [pair["frame_id_pair"][1] for pair in val_pairs[take_name]])))
    else:
        raise Exception("Take name not found in train or val pairs")
    # print(take_name, source_frame_id, target_frame_id)
    
    imu_csv_path = os.path.join(imu_file_path, take_name, "trajectory", "open_loop_trajectory.csv")
    
    imu_data = []
    prev_valid_imu = None
    with open(imu_csv_path, 'r') as csvfile:
        idx = 0
        if take_name not in imu_data_dict:
            imu_data_dict[take_name] = {}
            imu_output_dict[take_name] = {}

            csvreader = csv.DictReader(csvfile)
            prev_frame = None
            starting_frame = None
            for i, row in enumerate(csvreader):
                # if i % 1000 != 500:
                #     continue
                # if i == 500:
                #     starting_sec = int(row["tracking_timestamp_us"]) // 1e6
                
                second = int(row["tracking_timestamp_us"]) // 1e6 
                frame = int(row["tracking_timestamp_us"]) // (1e6 / 30)
                
                if starting_frame is None:
                    starting_frame = frame
                
                frame -= starting_frame
                 
                if prev_frame is None:
                    prev_frame = frame
                if prev_frame == frame:
                    prev_frame = frame
                    continue

                if frame not in frame_list:
                    continue

                    
                #print("SECONDS", second, prev_sec)
                # print("FRAMES", frame,prev_frame)

               # prev_frame = frame
                

                filtered_row = {
                    k: float(v) for k, v in row.items() 
                    if k not in ["tracking_timestamp_us", "utc_timestamp_ns", "session_uid", "quality_score"] and not k.startswith("gravity_")
                }
                imu_data.append(filtered_row)

                # frame_id = int(row["tracking_timestamp_us"]) // (1e6 / 30) - starting_sec * 30
                # print(second, frame_id)
                
                imu_info = list(filtered_row.values())
                sorted_keys = sorted(filtered_row.keys())
                imu_info = [filtered_row[k] for k in sorted_keys]
                
                imu_data_dict[take_name][frame_list[idx]] = imu_info
                if frame != prev_frame:
                    idx += 1
                    
                prev_frame = frame
            
            print(take_name, len(imu_data_dict[take_name]), len(frame_list))
            #raise Exception(i, take_name, len(imu_data_dict[take_name]))

        if source_frame_id not in imu_data_dict[take_name]:
            source_frame_imu = prev_valid_imu
        else:
            source_frame_imu = imu_data_dict[take_name][source_frame_id]
            prev_valid_imu = source_frame_imu

        if target_frame_id not in imu_data_dict[take_name]:
            print("Missing target frame:", take_name, target_frame_id)
            target_frame_imu = source_frame_imu
        else:
            target_frame_imu = imu_data_dict[take_name][target_frame_id]
            prev_valid_imu = target_frame_imu

    # print(imu_data_dict)
    # print(take_name, source_frame_id, target_frame_id)
    
    imu_output_dict[take_name][source_frame_id] = source_frame_imu
    imu_output_dict[take_name][target_frame_id] = target_frame_imu
    


print(imu_output_dict)
save_imu_output_path = "imu_data/egoexo4d_imu_data.json"
os.makedirs(os.path.dirname(save_imu_output_path), exist_ok=True)
with open(save_imu_output_path, "w") as f:
    json.dump(imu_output_dict, f)
print(f"Saved IMU data to {save_imu_output_path}")
                
    