import os
from glob import glob
import json
import numpy as np
from pprint import pprint

queried_frame_pairs = [("cmu_soccer06_2", 285, 315),
("unc_basketball_02-24-23_01_2",1845,1875  ),
("uniandes_bouldering_012_3", 975, 1005 ),
("indiana_cooking_09_2",75, 105),
("georgiatech_bike_07_6", 15, 45),
("minnesota_rockclimbing_032_2", 45, 75),
("uniandes_dance_016_37", 1575, 1605)
] # unc_basketball_02-24-23_01_2_001155_8779_0.jpg

pose_dict = {
    "type": "c2w",
    "frames": []
}

relative_pose_dict = {
    "type": "c2w",
    "frames": []
}


for take_name, source_frame_id, target_frame_id in queried_frame_pairs:
    # TODO: find source and target matrix that have the string '{take_name}_{source_frame_id:06d}' and '{take_name}_{target_frame_id:06d}'
    # note: the name is not simply just {take_name}_{source_frame_id:06d}, there are extra numbers after that.
    cwd = os.getcwd()
   ## print(cwd)

    source_path_search_query = os.path.join(
        cwd, "frame_pairs", take_name, f"{take_name}_{source_frame_id:06d}_source_matrix.npy"
    )
 #   print(source_path_search_query)
    source_matches = glob(source_path_search_query)
    if not source_matches:
        raise FileNotFoundError(f"No source match: {source_path_search_query}")
    source_matrix_path = source_matches[0]
    print(source_matrix_path)

    target_path_search_query = os.path.join(
        cwd, "frame_pairs", take_name, f"{take_name}_{target_frame_id:06d}_target_matrix.npy"
    )
   ## print(target_path_search_query)
    target_matches = glob(target_path_search_query)
    if not target_matches:
        raise FileNotFoundError(f"No target match: {target_path_search_query}")
    target_matrix_path = target_matches[0]
    print(target_matrix_path)


    source_matrix = np.load(source_matrix_path)
    target_matrix = np.load(target_matrix_path)

    source_image_path_search_query = os.path.join(
        cwd, "frame_pairs", take_name, f"{take_name}_{source_frame_id:06d}_*.jpg"
    )
    source_image_matches = glob(source_image_path_search_query)
    if not source_image_matches:
        raise FileNotFoundError(f"No source image match: {source_image_path_search_query}")
    source_image_path = source_image_matches[0]

    target_image_path_search_query = os.path.join(
        cwd, "frame_pairs", take_name, f"{take_name}_{target_frame_id:06d}_*.jpg"
    )
    target_image_matches = glob(target_image_path_search_query)
    if not target_image_matches:
        raise FileNotFoundError(f"No target image match: {target_image_path_search_query}")
    target_image_path= target_image_matches[0]

    print(f"Source image path: {source_image_path}")
    print(f"Target image path: {target_image_path}")

    # pose_dict["frames"].append({
    #     "image_name": os.path.basename(source_image_path),
    #     "pose": source_matrix.tolist()
    # })
    # pose_dict["frames"].append({
    #     "image_name": os.path.basename(target_image_path),
    #     "pose": target_matrix.tolist()
    # })


    relative_pose_matrix = np.linalg.inv( source_matrix) @ target_matrix
    relative_pose_path_search_query = os.path.join(
        cwd, "frame_pairs", take_name, f"predicted_{take_name}_{source_frame_id:06d}_to_{take_name}_{target_frame_id:06d}.npy"
    )
    relative_pose_pred_path = glob(relative_pose_path_search_query)[0]
    if not relative_pose_pred_path:
        raise FileNotFoundError(f"No relative pose match: {relative_pose_path_search_query}")
    relative_pose_matrix_pred = np.load(relative_pose_pred_path).squeeze()


    our_relative_pose_path_search_query = os.path.join(
        cwd, "frame_pairs", take_name, f'predicted_ours_{take_name}_{source_frame_id:06d}_to_{take_name}_{target_frame_id:06d}.npy'
    )
    our_relative_pose_pred_matches_path = glob(our_relative_pose_path_search_query)[0]
    if not our_relative_pose_pred_matches_path:
        raise FileNotFoundError(f"No relative pose match: {our_relative_pose_path_search_query}")
    our_relative_pose_matrix_pred = np.load(our_relative_pose_pred_matches_path).squeeze()

    # print(relative_pose_pred_path)
    # print(our_relative_pose_pred_matches_path)
    # # print(relative_pose_matrix)
    # print(relative_pose_matrix_pred)
    # print(our_relative_pose_matrix_pred)

    baseline_target_matrix = source_matrix @ relative_pose_matrix_pred
    our_target_matrix = source_matrix @ our_relative_pose_matrix_pred

    baseline_target_matrix_path_name = os.path.join(os.path.basename(source_image_path)).replace(".jpg", "_baseline.npy")
    our_target_matrix_path_name = os.path.join(os.path.basename(target_image_path)).replace(".jpg", "_ours.npy")

    os.makedirs("visualize_source_target_matrices/", exist_ok=True)
    np.save(os.path.join("visualize_source_target_matrices/", baseline_target_matrix_path_name), baseline_target_matrix)
    np.save(os.path.join("visualize_source_target_matrices/", our_target_matrix_path_name), our_target_matrix)
    np.save(os.path.join("visualize_source_target_matrices/", os.path.basename(source_image_path).replace(".jpg", "_source_matrix.npy")), source_matrix)
    np.save(os.path.join("visualize_source_target_matrices/", os.path.basename(target_image_path).replace(".jpg", "_target_matrix.npy")), target_matrix)
    pose_dict["frames"].append({
        "image_name": baseline_target_matrix_path_name,
        "pose": (source_matrix @ relative_pose_matrix_pred).tolist()
    })
    pose_dict["frames"].append({
        "image_name": our_target_matrix_path_name,
        "pose": (source_matrix @ our_relative_pose_matrix_pred).tolist()
    })
    pose_dict["frames"].append({
        "image_name": os.path.basename(source_image_path).replace(".jpg", "_source_matrix.npy"),
        "pose": source_matrix.tolist()
    })
    pose_dict["frames"].append({
        "image_name": os.path.basename(target_image_path).replace(".jpg", "_target_matrix.npy"),
        "pose": target_matrix.tolist()
    })

    print(np.linalg.norm(relative_pose_matrix - relative_pose_matrix_pred))
    print(np.linalg.norm(relative_pose_matrix - our_relative_pose_matrix_pred))

    # save relative_pose_matrix as .npy based on source and target frame_ids
   # print(relative_pose_dict)
    our_relative_pose_matrix_pred = our_relative_pose_matrix_pred.squeeze()
    relative_pose_matrix_pred = relative_pose_matrix_pred.squeeze()
    print(relative_pose_matrix.shape, source_matrix.shape, our_relative_pose_matrix_pred.shape, relative_pose_matrix_pred.shape)

    gt_pose_path_name = os.path.basename(relative_pose_pred_path).replace("predicted", "gt")
    baseline_pose_path_name = os.path.basename(relative_pose_pred_path).replace("predicted", "baseline")
    ours_pose_path_name = os.path.basename(relative_pose_pred_path).replace("predicted", "ours")

    np.save(gt_pose_path_name, relative_pose_matrix)
    np.save(baseline_pose_path_name, relative_pose_matrix_pred)
    np.save(ours_pose_path_name, our_relative_pose_matrix_pred)


    print(gt_pose_path_name)
    print(baseline_pose_path_name)
    print(ours_pose_path_name)
    relative_pose_dict["frames"].append({
        "image_name": gt_pose_path_name.replace(".npy", ".jpg"),
        "pose": relative_pose_matrix.tolist()
    })
    relative_pose_dict["frames"].append({
        "image_name": baseline_pose_path_name.replace(".npy", ".jpg"),
        "pose": relative_pose_matrix_pred.tolist()
    })
    relative_pose_dict["frames"].append({
        "image_name": ours_pose_path_name.replace(".npy", ".jpg"),
        "pose": our_relative_pose_matrix_pred.tolist()
    })

# save pose_dict as json
with open("visualized_pose_info.json", "w") as f:
    json.dump(pose_dict, f, indent=4)

# # save relative_pose_dict as json   
# with open("visualized_relative_pose_info.json", "w") as f:
#     json.dump(relative_pose_dict, f, indent=4)