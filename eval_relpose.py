import argparse
import os
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from reloc3r.reloc3r_relpose import Reloc3rRelpose, setup_reloc3r_relpose_model, inference_relpose
from reloc3r.datasets import get_data_loader
from reloc3r.utils.metric import *
from reloc3r.utils.device import to_numpy

from tqdm import tqdm
import json
# from pdb import set_trace as bb
from pprint import pprint

def get_args_parser():
    parser = argparse.ArgumentParser(description='evaluation code for relative camera pose estimation')

    # model
    parser.add_argument('--model', type=str, 
        # default='Reloc3rRelpose(img_size=224)')
        default='Reloc3rRelpose(img_size=512)')
    
    # test set
    parser.add_argument('--test_dataset', type=str, 
        # default="ScanNet1500(resolution=(224,224), seed=777)")
        default="ScanNet1500(resolution=(512,384), seed=777)")
    parser.add_argument('--batch_size', type=int,
        default=1)
    parser.add_argument('--num_workers', type=int,
        default=10)
    parser.add_argument('--sound_length', type=int, default=60, 
        help='length of sound in milliseconds, used for audio-based relative pose estimation')
    parser.add_argument('--amp', type=int, default=1,
                                choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument('--rot_only', action='store_true',
        help='if set, only rotation error will be computed, translation error will be set to 0')
    # parser.add_argument('--output_dir', type=str, 
    #     default='./output', help='path where to save the pose errors')

    return parser


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print('Building {} data loader for {}'.format(split, dataset))
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))
    print('Dataset length: ', len(loader))
    return loader


def test(args):
    
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    device ='cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    reloc3r_relpose = setup_reloc3r_relpose_model(args.model, device)
    
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    # start evaluation
    rerrs, terrs = [], []
    rerrs_per_video = {}
    terrs_per_video = {}

    rerrs_per_angle_range = {}
    terrs_per_angle_range = {}

    # START MEAN CALCS
    # Calculate mean poses for baseline comparison
    mean_rotation_errors = []
    mean_translation_errors = []
    all_gt_poses = []

    calc_mean_info = False
    bin_total = 0
    # First pass: collect all ground truth poses to calculate means
    audio_and_visual_info = json.load(open("ssim_info/val_frame_pairs_ssim_info.json"))
    if calc_mean_info:
        print("Collecting ground truth poses for mean calculation...")
        for test_name, testset in data_loader_test.items():
            with torch.no_grad():
                for batch in tqdm(testset, desc=f"Collecting GT poses from {test_name}"):
                    view1, view2 = batch
                    # if view1["dataset"][0].lower().startswith("slfm"):
                    #     gt_pose2to1 = view2['camera_pose'] - view1['camera_pose']
                    #     all_gt_poses.append(gt_pose2to1)
                    #     continue  # Skip SLfM as it doesn't have camera poses
                    gt_pose2to1 = torch.inverse(view1['camera_pose']) @ view2['camera_pose']
                    all_gt_poses.append(gt_pose2to1)

        # Calculate mean rotation and translation
        all_gt_poses = torch.cat(all_gt_poses, dim=0)
        mean_rotation = torch.mean(all_gt_poses[:, 0:3, 0:3], dim=0)
        mean_translation = torch.mean(all_gt_poses[:, 0:3, 3], dim=0)

        print(f"Mean rotation matrix calculated from {len(all_gt_poses)} poses")
        print(f"Mean translation vector calculated from {len(all_gt_poses)} poses")

        # Create mean pose prediction for comparison
        mean_pose_pred = torch.zeros_like(all_gt_poses[0:1]).repeat(1, 1, 1)
        mean_pose_pred[0, 0:3, 0:3] = mean_rotation
        mean_pose_pred[0, 0:3, 3] = mean_translation
        mean_pose_pred[0, 3, 3] = 1.0

        print("Mean pose baseline prediction created")
    # END MEAN CALCS


    mean_pose_pred = torch.tensor([[[ 8.9121e-01,  3.3189e-03, -8.3057e-04, -1.1656e-02],
        [-3.0662e-03,  8.9137e-01,  3.2967e-04, -9.2079e-03],
        [-1.7996e-04,  1.0292e-03,  9.7111e-01,  4.6023e-03],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]])

    # for each frame pair keep track of the best results
    result_tracker = {}
    ground_truth_pose_tracker = {}
    policy_num_correct = 0
    policy_total = 0
    count_audio_usage = 0
    count_visual_usage = 0
    bin_acc = 0
    min_angle_range = np.inf
    max_angle_range = -np.inf

    sample_cutoff = 10000
    current_sample_id = 0
    min_transl_mag = np.inf
    max_transl_mag = -np.inf
    transl_mags = []
    n = 0
    
    rerr_high_translation = []
    terr_high_translation = []
    
    rerr_low_camera_motion_high_audio_source_change = []
    terr_low_camera_motion_high_audio_source_change = []
    rerr_high_camera_motion_low_audio_source_change = []
    terr_high_camera_motion_low_audio_source_change = []
    
    rerr_decoupled_audio_camera_motion = []
    terr_decoupled_audio_camera_motion = []
    
    rerr_high_camera_motion_high_audio_source_change = []
    terr_high_camera_motion_high_audio_source_change = []
    
    for test_name, testset in data_loader_test.items():
        print('Testing {:s}'.format(test_name))
        with torch.no_grad():
            
            for batch in tqdm(testset):
                # if current_sample_id >= sample_cutoff:
                #     break
                pose = inference_relpose(batch, reloc3r_relpose, device, use_amp=bool(args.amp))

                view1, view2 = batch
                # if "slfm" in view1["dataset"][0].lower():
                #     rerrs_prh = []
                #     terrs_prh = []

                #     gt_pose2to1 = abs(view1["camera_pose"] - view2["camera_pose"])
                #     gt_pose2to1 = (gt_pose2to1 + 180) % 360 - 180
                #     #print(gt_pose2to1, pose)
                #     for sid in range(len(pose)):
                #         # rerr = abs(gt_pose2to1[sid] - pose[sid])

                #                 #rerrs_prh.append(rerr) 
                #         rerrs.append(rerr.cpu())
                #         source_label = view1["label"][sid]
                #         target_label = view2["label"][sid]
                #         result_tracker[(source_label, target_label)] = {
                #             "rotation_error": rerr.cpu(),
                #             "translation_error": 0
                #         }
                #     continue

                if "doas" in view1:
                    doa = view1["doas"]
                    pred_doa = pose
                    
                    doa_sample = doa[0]
                    label = view1["label"][0]

                    pred_doa_sample = pred_doa[0]
                    # import matplotlib.pyplot as plt

                    # plt.figure(figsize=(8, 4))
                    # plt.plot(doa_sample.cpu().numpy(), label='GT DOA')
                    # plt.plot(pred_doa_sample.cpu().numpy(), label=f'Predicted DOA {label}')
                    # plt.legend()
                    # plt.title('DOA Comparison')
                    # plt.xlabel('Index')
                    # plt.ylabel('DOA Value')
                    # plt.tight_layout()
                    # plot_path = 'doa_plots_egoexo4d_alt'
                    # if not os.path.exists(plot_path):
                    #     os.makedirs(plot_path)
                    # plt.savefig(f'{plot_path}/doa_comparison_{label}.png')
                    # plt.close()
                    # raise Exception("DOA comparison plot saved, stopping execution for inspection.")
                elif "policy_decision" in view1:
                    policy_decision = view1["policy_decision"]
                    pred_decision = pose
                    # Vectorized accuracy computation
                    target = policy_decision
                    pred_flat = pred_decision.view(-1)
                    target_flat = target.view(-1)
                    policy_num_correct += (pred_flat == target_flat).sum().item()
                    policy_total += pred_flat.numel()
                
                    for sid in range(len(pose)):
                    
                        if pred_decision[sid] == 0:
                            rerr = view1["vision_rot_error"][sid].item()
                            terr = view1["vision_trans_error"][sid].item()
                            count_visual_usage += 1
                        else:
                            rerr = view1["audio_rot_error"][sid].item()
                            terr = view1["audio_trans_error"][sid].item()
                            count_audio_usage += 1
                        
                        rerrs.append(rerr)
                        terrs.append(terr)
                elif "waveform" in view1:
                    output = pose
                else:
                    gt_pose2to1 = torch.inverse(view1['camera_pose']) @ view2['camera_pose']
                    rerrs_prh = []
                    terrs_prh = []
                    
                    # rotation angular err
                    R_prd = pose[:,0:3,0:3]
                    current_sample_id += len(R_prd)
                    #  print(current_sample_id)
                    for sid in range(len(R_prd)):
                        rerr = get_rot_err(to_numpy(R_prd[sid]), to_numpy(gt_pose2to1[sid,0:3,0:3]))
                        rerrs_prh.append(rerr)
                        
                        if "slfm" in view1["dataset"][0].lower():
                            # convert rot matrix into angle in degrees
                            gt_rot = gt_pose2to1[sid,0:3,0:3]
                            gt_rot_angle = np.arccos((np.trace(to_numpy(gt_rot)) - 1) / 2) * 180.0 / np.pi
                            
                            pred_rot = R_prd[sid]
                            pred_rot_angle = np.arccos((np.trace(to_numpy(pred_rot)) - 1) / 2) * 180.0 / np.pi

                            gt_rot_angle = (gt_rot_angle + 180) % 360 - 180
                            pred_rot_angle = (pred_rot_angle + 180) % 360 - 180

                            angle_range = 360
                            num_classes = 64
                            bin_size = angle_range / (num_classes // 2)
                            angle_min = -angle_range / 2 
                            angle_max = angle_range / 2 
                            if gt_rot_angle >= 0:
                                gt_angle_bin = (gt_rot_angle - angle_min) // bin_size
                                gt_angle_bin = np.clip(gt_angle_bin, 0, num_classes // 2 - 1) + num_classes // 2
                            elif gt_rot_angle < 0:
                                gt_angle_bin = (gt_rot_angle + angle_max) // bin_size
                                gt_angle_bin = np.clip(gt_angle_bin, 0, num_classes // 2 - 1)

                            if pred_rot_angle >= 0:
                                pred_angle_bin = (pred_rot_angle - angle_min) // bin_size
                                pred_angle_bin = np.clip(pred_angle_bin, 0, num_classes // 2 - 1) + num_classes // 2
                            elif pred_rot_angle < 0:
                                pred_angle_bin = (pred_rot_angle + angle_max) // bin_size
                                pred_angle_bin = np.clip(pred_angle_bin, 0, num_classes // 2 - 1)

                            bin_acc += (gt_angle_bin == pred_angle_bin)

                        # # begin mean
                        # rerr = get_rot_err(to_numpy(mean_pose_pred[0, 0:3, 0:3]), to_numpy(gt_pose2to1[sid,0:3,0:3]))
                        # rerrs_prh.append(rerr)
                        # end mean

                        if "slfm" not in view1["dataset"][0].lower():

                            take_name = view1["take_name"] [sid]
                            source_frame_id = view1["source_frame_id"][sid].item()
                            target_frame_id = view2["target_frame_id"][sid].item()

                            result_tracker[take_name,source_frame_id,target_frame_id] = {
                                "rotation_error": rerr
                            }
                        else:
                            source_label = view1["label"][sid]
                            target_label = view2["label"][sid]
                            rerr = get_rot_err(to_numpy(R_prd[sid]), to_numpy(gt_pose2to1[sid,0:3,0:3]))
                            result_tracker[(source_label, target_label)] = {
                                "rotation_error": rerr,
                                "source_img_path": view1["img_path"][sid],
                                "target_img_path": view2["img_path"][sid],
                                "source_sound_path": view1["sound_path"][sid],
                                "target_sound_path": view2["sound_path"][sid],
                                "translation_error": 0
                            }
                            if "sound_path" in view1:
                                result_tracker[(source_label, target_label)]["sound_path"] = view1["sound_path"][sid]
                            

                    if "slfm" in view1["dataset"][0].lower():
                        rerrs += rerrs_prh
                        continue

                        #rerrs_prh.append(get_rot_err(to_numpy(mean_pose_pred[0, 0:3, 0:3]), to_numpy(gt_pose2to1[sid,0:3,0:3])))
                    #     rand_r = torch.randn(3, 3)
                    #     # Make the random matrix orthogonal using QR decomposition
                    #     rand_r, _ = torch.linalg.qr(rand_r)
                    #     # Ensure proper rotation matrix (det = 1)
                    #     if torch.det(rand_r) < 0:
                    #         rand_r[:, 0] *= -1
                    #     rerrs_prh.append(get_rot_err(to_numpy(rand_r), to_numpy(gt_pose2to1[sid,0:3,0:3])))

                    # translation direction angular err
                    t_prd = pose[:,0:3,3]
                    for sid in range(len(t_prd)): 
                        transl = to_numpy(t_prd[sid])


                        # begin mean
                        # transl = to_numpy(mean_pose_pred[0, 0:3, 3])
                        # # end mean

                        gt_transl = to_numpy(gt_pose2to1[sid,0:3,-1])
                        transl_dir = transl / np.linalg.norm(transl)
                        gt_transl_dir = gt_transl / np.linalg.norm(gt_transl)
                        terr = get_transl_ang_err(transl_dir, gt_transl_dir)
                        terrs_prh.append(terr)

                        if "slfm" not in view1["dataset"][0].lower():
                            take_name = view1["take_name"] [sid]
                            source_frame_id = view1["source_frame_id"][sid].item()
                            target_frame_id = view2["target_frame_id"][sid].item()
                            result_tracker[take_name,source_frame_id,target_frame_id]["translation_error"] = terr

                        

                    rerrs += rerrs_prh
                    terrs += terrs_prh

                    # for video in view1["take_name"]:
                    #     # TODO: calculate and save pose errors for each video and print results
                    #     rerrs_per_video.setdefault(video, []).extend(rerrs_prh)
                    #     terrs_per_video.setdefault(video, []).extend(terrs_prh)

                    # group aucs by rotation and translation differences (determine auc for small relative camera pose ground truths, medium and large, but provide 4 ranges instead of 3)
                    for sid in range(len(rerrs_prh)):
                        # Calculate ground truth rotation and translation magnitudes
                        gt_R = to_numpy(gt_pose2to1[sid, 0:3, 0:3])
                        gt_t = to_numpy(gt_pose2to1[sid, 0:3, 3])
                        
                        gt_rot_angle = get_rot_err(gt_R, np.eye(3))  # rotation angle from identity
                        gt_transl_magnitude = np.linalg.norm(gt_t)  # translation magnitude
                        #  gt_transl_dir = gt_t / gt_transl_magnitude if gt_transl_magnitude > 0 else gt_t

                        # take_name = view1["take_name"] [sid]
                        # source_frame_id = view1["source_frame_id"][sid].item()
                        # target_frame_id = view2["target_frame_id"][sid].item()

                        # TODO: save these ground truth values if needed
                        # ground_truth_pose_tracker[take_name,source_frame_id,target_frame_id] = {
                        #     "gt_rot": float(gt_rot_angle),
                        #     "gt_trans_mag": float(gt_transl_magnitude),
                        # }   

                        # Group by ground truth rotation angle ranges
                        if gt_rot_angle < 5:
                            rerrs_per_angle_range.setdefault('0-5', []).append(rerrs_prh[sid])
                            terrs_per_angle_range.setdefault('0-5', []).append(terrs_prh[sid])
                        elif gt_rot_angle < 10:
                            rerrs_per_angle_range.setdefault('5-10', []).append(rerrs_prh[sid])
                            terrs_per_angle_range.setdefault('5-10', []).append(terrs_prh[sid])
                        elif gt_rot_angle < 20:
                            rerrs_per_angle_range.setdefault('10-20', []).append(rerrs_prh[sid])
                            terrs_per_angle_range.setdefault('10-20', []).append(terrs_prh[sid])
                        else:
                            rerrs_per_angle_range.setdefault('20+', []).append(rerrs_prh[sid])
                            terrs_per_angle_range.setdefault('20+', []).append(terrs_prh[sid])
                            
                        # min_transl_mag = min(gt_transl_magnitude, min_transl_mag)
                        # max_transl_mag = max(gt_transl_magnitude, max_transl_mag)
                        # transl_mags.append(gt_transl_magnitude)
                        
                        # translation_mag_mean = 1.1367295
                        # translation_mag_std = 0.5120323
                        # if gt_transl_magnitude > translation_mag_mean and rerrs_prh[sid] < 10: # TODO: also account for low rotation
                        #     rerr_high_translation.append(rerrs_prh[sid])
                        #     terr_high_translation.append(terrs_prh[sid])
                        
                        take_name = view1["take_name"] [sid]
                        source_frame_id = view1["source_frame_id"][sid].item()
                        target_frame_id = view2["target_frame_id"][sid].item()
                        
                        ssim = audio_and_visual_info[str((take_name,source_frame_id,target_frame_id))]["similarity"]
                        audio_diff = audio_and_visual_info[str((take_name, source_frame_id, target_frame_id))]["log_mel_intensity_difference"]  
                    
                        # if rerrs_prh[sid] < 10 and audio_diff > 0.18: # ssim > 0.7 and audio_diff > 0.18:
                        #     rerr_low_camera_motion_high_audio_source_change.append(rerrs_prh[sid])
                        #     terr_low_camera_motion_high_audio_source_change.append(terrs_prh[sid])    
                        
                        # if ssim < 0.4 and audio_diff < 0.04:
                        #     rerr_high_camera_motion_low_audio_source_change.append(rerrs_prh[sid])
                        #     terr_high_camera_motion_low_audio_source_change.append(terrs_prh[sid])
                        
                        # if (rerrs_prh[sid] < 10 and audio_diff > 0.18) or (ssim < 0.4 and audio_diff < 0.04):
                        #     rerr_decoupled_audio_camera_motion.append(rerrs_prh[sid])
                        #     terr_decoupled_audio_camera_motion.append(terrs_prh[sid])
                        
                        if gt_rot_angle > 20 and audio_diff > 0.18:
                            rerr_high_camera_motion_high_audio_source_change.append(rerrs_prh[sid])
                            terr_high_camera_motion_high_audio_source_change.append(terrs_prh[sid])
                            
        if policy_total > 0:
            print("Classifier Accuracy", policy_num_correct/ policy_total)
            print("Audio usage count", count_audio_usage)
            print("Visual usage count", count_visual_usage)
            
        # mean_transl_mag = np.mean(transl_mags)
        # std_transl_mag =  np.std(transl_mags)
        # med_transl_mag = np.median(transl_mags)
        # transl_mag_first_quartile = np.percentile(transl_mags, 25)
        # transl_mag_third_quartile = np.percentile(transl_mags, 75)
        # print("TRANSLATION MIN and MAX", min_transl_mag, max_transl_mag)
        # print("TRANSLATION STD and MEAN", std_transl_mag, mean_transl_mag)
        # print("1ST Quartile, MEDIAN, 3RD Quartile", transl_mag_first_quartile, med_transl_mag, transl_mag_third_quartile)

        # print per video results 
        rerrs = np.array(rerrs)
        terrs = np.array(terrs)
        
        # frame_pair_result_tracker_file = "frame_pair_pose_errors_vision_and_doa_hm3d.npy"
        # np.save(frame_pair_result_tracker_file, result_tracker)
        
        # print("Rotation error (mean):", np.mean(rerrs))

        # raise Exception("done")
        # rerr_high_translation = np.array(rerr_high_translation)
        # terr_high_translation = np.array(terr_high_translation)
        # rerr_low_camera_motion_high_audio_source_change = np.array(rerr_low_camera_motion_high_audio_source_change)
        # terr_low_camera_motion_high_audio_source_change = np.array(terr_low_camera_motion_high_audio_source_change)
        
        # rerr_high_camera_motion_low_audio_source_change = np.array(rerr_high_camera_motion_low_audio_source_change)
        # terr_high_camera_motion_low_audio_source_change = np.array(terr_high_camera_motion_low_audio_source_change)
       
        # rerr_high_camera_motion_high_audio_source_change = np.array(rerr_high_camera_motion_high_audio_source_change)
        # terr_high_camera_motion_high_audio_source_change = np.array(terr_high_camera_motion_high_audio_source_change)
        # print('In total {} pairs'.format(len(rerr_high_camera_motion_high_audio_source_change)))
        
        # auc
        print("TOTAL AUCs")
        print(error_auc(rerrs, terrs, thresholds=[5, 10, 20]))
        print("ROTATION ONLY AUCs")
        print(error_auc(rerrs, np.zeros_like(terrs), thresholds=[5, 10, 20]))
        print("TRANSLATION ONLY AUCs")
        print(error_auc(np.zeros_like(rerrs), terrs, thresholds=[5, 10, 20]))

        print("MEAN ERROR")
        print("rotation", np.mean(rerrs))
        print("translation", np.mean(terrs))
        print("Median Error")
        print("rotation", np.median(rerrs))
        print("translation", np.median(terrs))
        
        raise  Exception("done")
        
        # print("HIGH CAMERA MOTION HIGH AUDIO SOURCE CHANGE TOTAL AUCs")
        # print(error_auc(rerr_high_camera_motion_high_audio_source_change, terr_high_camera_motion_high_audio_source_change, thresholds=[5, 10, 20]))
        # print("HIGH CAMERA MOTION HIGH AUDIO SOURCE CHANGE ROT ONLY AUC")
        # print(error_auc(rerr_high_camera_motion_high_audio_source_change, np.zeros_like(terr_high_camera_motion_high_audio_source_change), thresholds=[5, 10,20]))
        # print("HIGH CAMERA MOTION HIGH AUDIO SOURCE CHANGE TRANS ONLY AUC")
        # print(error_auc(np.zeros_like(rerr_high_camera_motion_high_audio_source_change), terr_high_camera_motion_high_audio_source_change,  thresholds=[5, 10,20]))
        
        # rerr_decoupled_audio_camera_motion = np.array(rerr_decoupled_audio_camera_motion)
        # terr_decoupled_audio_camera_motion = np.array(terr_decoupled_audio_camera_motion)
        # print("DECOUPLED AUDIO-CAMERA MOTION TOTAL AUCs")
        # print(error_auc(rerr_decoupled_audio_camera_motion, terr_decoupled_audio_camera_motion, thresholds=[5, 10, 20]))
        # print("DECOUPLED AUDIO-CAMERA MOTION ROT ONLY AUC")
        # print(error_auc(rerr_decoupled_audio_camera_motion, np.zeros_like(terr_decoupled_audio_camera_motion), thresholds=[5, 10,20]))
        # print("DECOUPLED AUDIO-CAMERA MOTION TRANS ONLY AUC")
        # print(error_auc(np.zeros_like(rerr_decoupled_audio_camera_motion), terr_decoupled_audio_camera_motion,  thresholds=[5, 10,20]))
        
        # print("LOW CAMERA MOTION HIGH AUDIO SOURCE CHANGE TOTAL AUCs")
        # print(error_auc(rerr_low_camera_motion_high_audio_source_change, terr_low_camera_motion_high_audio_source_change, thresholds=[5, 10, 20]))
        # print("LOW CAMERA MOTION HIGH AUDIO SOURCE CHANGE ROT ONLY AUC")
        # print(error_auc(rerr_low_camera_motion_high_audio_source_change, np.zeros_like(terr_low_camera_motion_high_audio_source_change), thresholds=[5, 10,20]))
        # print("LOW CAMERA MOTION HIGH AUDIO SOURCE CHANGE TRANS ONLY AUC")
        # print(error_auc(np.zeros_like(rerr_low_camera_motion_high_audio_source_change), terr_low_camera_motion_high_audio_source_change,  thresholds=[5, 10,20]))
        
        # print("HIGH CAMERA MOTION LOW AUDIO SOURCE CHANGE TOTAL AUCs")
        # print(error_auc(rerr_high_camera_motion_low_audio_source_change, terr_high_camera_motion_low_audio_source_change, thresholds=[5, 10, 20]))
        # print("HIGH CAMERA MOTION LOW AUDIO SOURCE CHANGE ROT ONLY AUC")
        # print(error_auc(rerr_high_camera_motion_low_audio_source_change, np.zeros_like(terr_high_camera_motion_low_audio_source_change), thresholds=[5, 10,20]))
        # print("HIGH CAMERA MOTION LOW AUDIO SOURCE CHANGE TRANS ONLY AUC")
        # print(error_auc(np.zeros_like(rerr_high_camera_motion_low_audio_source_change), terr_high_camera_motion_low_audio_source_change,  thresholds=[5, 10,20]))
        

        # print("HIGH TRANSLATION LOW ROTATION TOTAL AUCs")
        # print(error_auc(rerr_high_translation, terr_high_translation, thresholds=[5, 10, 20]))
        # print("HIGH TRANSLATION LOW ROTATION ROT ONLY AUC")
        # print(error_auc(rerr_high_translation, np.zeros_like(terr_high_translation), thresholds=[5, 10, 20]))
        # print("HIGH TRANSLATION LOW ROTATION TRANS ONLY AUC")
        # print(error_auc(np.zeros_like(rerr_high_translation), terr_high_translation, thresholds=[5, 10, 20]))
        # print(len(rerr_high_translation), "samples with high translation magnitude (above mean)")
        
        # frame_pair_result_tracker_file = "frame_pair_pose_errors_doa_slfm_cp_1000ms_TRAIN.npy"
        # np.save(frame_pair_result_tracker_file, result_tracker)
        # print(f'Frame pair pose errors saved to {frame_pair_result_tracker_file}')

        #frame_pair_result_tracker_file = "frame_pair_pose_errors_slfm_and_doa_360.npy"
        # frame_pair_result_tracker_file = "frame_pair_pose_errors_corrupted_slfm_and_doa_360_1000ms_TRAIN.npy"
        # np.save(frame_pair_result_tracker_file, result_tracker)
        # print(f'Frame pair pose errors saved to {frame_pair_result_tracker_file}')


        
        # save angle range aucs to json
        angle_range_results = {}
        for angle_range in rerrs_per_angle_range.keys():
            rerrs_np = np.array(rerrs_per_angle_range[angle_range])
            terrs_np = np.array(terrs_per_angle_range[angle_range])

            auc_errors = error_auc(rerrs_np, terrs_np, thresholds=[5, 10, 20])
            auc_rotation_only_errors = error_auc(rerrs_np, np.zeros_like(terrs_np), thresholds=[5, 10, 20])
            auc_translation_only_errors = error_auc(np.zeros_like(rerrs_np), terrs_np, thresholds=[5, 10, 20])
            angle_range_results[angle_range] = {"total_aucs": auc_errors, "rotation_only_aucs": auc_rotation_only_errors, "translation_only_aucs": auc_translation_only_errors}
        print("Angle Range AUCs:")
        # pprint(angle_range_results) 
        # # save angle range results to json file
        # angle_range_output_file = f'angle_range_auc_results_vision_only_NEW.json' #f'angle_range_auc_results_vision_only.json' # 
        # with open(angle_range_output_file, 'w') as f:
        #     json.dump(angle_range_results, f, indent=2)
        # print(f'Angle range AUC results saved to {angle_range_output_file}')

        # # save per-video results to json
        # per_video_results = {}
        # for video in rerrs_per_video.keys():
        #     rerrs_per_video_np = np.array(rerrs_per_video[video])
        #     terrs_per_video_np = np.array(terrs_per_video[video])

        #     auc_errors = error_auc(rerrs_per_video_np, terrs_per_video_np, thresholds=[5, 10, 20])
        #     auc_errors_rotation_only = error_auc(rerrs_per_video_np, np.zeros_like(terrs_per_video_np), thresholds=[5, 10, 20])
        #     auc_errors_translation_only = error_auc(np.zeros_like(rerrs_per_video_np), terrs_per_video_np, thresholds=[5, 10, 20])

        #    # print(f'Video: {video}, Total AUCs: {str(auc_errors)}, Rotation-Only AUCs: {str(auc_errors_rotation_only)}, Translation-Only AUCs: {str(auc_errors_translation_only)}')
        #     per_video_results[video] = {
        #         'total_aucs': auc_errors.tolist() if hasattr(auc_errors, 'tolist') else auc_errors,
        #         'rotation_only_aucs': auc_errors_rotation_only.tolist() if hasattr(auc_errors_rotation_only, 'tolist') else auc_errors_rotation_only,
        #         'translation_only_aucs': auc_errors_translation_only.tolist() if hasattr(auc_errors_translation_only, 'tolist') else auc_errors_translation_only
        #      }

        # print("total_vid_count", len(rerrs_per_video.keys()))
        
        # per_video_output_file = f'per_video_auc_results_vision_only.json' 
        # with open(per_video_output_file, 'w') as f:
        #     json.dump(per_video_results, f, indent=2)
        # print(f'Per-video AUC results saved to {per_video_output_file}')

        # with open("ground_truth_pose_diffs_train.json", 'w') as f:
        #     json.dump({str(k) : v for k, v in ground_truth_pose_tracker.items()}, f, indent=2)
        # print(f'Ground truth pose differences saved to ground_truth_pose_diffs_train.json')
        
        
        # # # # save results to json file
        # per_video_output_file = f'per_video_auc_results_doa_cp_1000ms_TRAIN.json' #'per_video_auc_results_vision_only.json' 
        # with open(per_video_output_file, 'w') as f:
        #     json.dump(per_video_results, f, indent=2)
        # print(f'Per-video AUC results saved to {per_video_output_file}')


                    # save printed information into a results folder
                    #output_file = '{}/{}_pose_error_results.json'.format(args.output_dir, video)
                    # if not os.path.exists(os.path.dirname(output_file)):
                    #     os.makedirs(os.path.dirname(output_file))
                        
                    # results = {
                    #     'video': video,
                    #     'rotation_errors': rerrs_per_video[video].tolist(),
                    #     'translation_errors': terrs_per_video[video].tolist(),
                    #     'total_aucs': error_auc(rerrs_per_video[video], terrs_per_video[video], thresholds=[5, 10, 20]),
                    #     'rotation_only_aucs': error_auc(terrs_per_video[video], np.zeros_like(terrs_per_video[video]), thresholds=[5, 10, 20])
                    # }

                    # with open(output_file, 'w') as f:
                    #     json.dump(results, f, indent=2)
                    
        
        # frame_pair_result_tracker_file = "frame_pair_pose_errors_slfm_and_doa_360_TRAIN.npy"
        # np.save(frame_pair_result_tracker_file, result_tracker)
        # print(f'Frame pair pose errors saved to {frame_pair_result_tracker_file}')
        # raise Exception("done")

        # # save err list to file
        # err_list = np.concatenate((rerrs[:,None], terrs[:,None]), axis=-1)
        # output_file = '{}/pose_error_list.txt'.format(args.output_dir)
        # np.savetxt(output_file, err_list)
        # print('Pose errors saved to {}'.format(output_file))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    test(args)

