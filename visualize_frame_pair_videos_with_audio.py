import os
import json
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import librosa
import librosa.display
import moviepy
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from moviepy.editor import TextClip, CompositeVideoClip
from pprint import pprint
from tqdm import tqdm
from moviepy.video.fx import all as vfx
from moviepy.editor import clips_array
import cv2

audio_file_dir = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/audio/" # '/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/audio/cmu_bike01_4/sound'
frame_dir = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/frames/"
doa_dir = "/vision/vision_data_2/egoexo4d_audio/camera_pose_audio_data/doa_data/" # 000015_duration_1000ms_azimuths.npy

camera_pose_pred_results_dir = "result_poses" # /ground_truths
output_visualization_dir = "visualize_HM3D-SS-frame_pair_videos_with_audio/"

def get_rot_err(rot_a, rot_b):
    rot_err = rot_a.T.dot(rot_b)
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return rot_err[0]

def get_transl_ang_err(dir_a, dir_b):
    dot_product = np.sum(dir_a * dir_b)
    cos_angle = dot_product / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    err = np.degrees(angle)
    return err


if __name__ == "__main__":
    # frame_pairs_vision_only = np.load("frame_pair_pose_errors_vision_only.npy", allow_pickle=True).item()
    # frame_pairs_ours = np.load("frame_pair_pose_errors_slfm_and_doa_360.npy", allow_pickle=True).item()
    frame_pairs_vision_only = np.load("frame_pair_pose_errors_vision_only_hm3d.npy", allow_pickle=True).item()
    frame_pairs_ours = np.load("frame_pair_pose_errors_vision_and_doa_hm3d.npy", allow_pickle=True).item()


    print(len(frame_pairs_vision_only.keys()), len(frame_pairs_ours.keys()))
    error_diffs = []
    for frame_pair_info in frame_pairs_vision_only.keys():
        source_label, target_label = frame_pair_info

        vision_only_errs = frame_pairs_vision_only[frame_pair_info]
        ours_errs = frame_pairs_ours[frame_pair_info]

        vision_only_rot_err = vision_only_errs["rotation_error"]
        ours_rot_err = ours_errs["rotation_error"]

        vision_only_trans_err = vision_only_errs["translation_error"]
        ours_trans_err = ours_errs["translation_error"]

        rot_diff = vision_only_rot_err - ours_rot_err
        trans_diff = vision_only_trans_err - ours_trans_err 
        both_better = 0
        if rot_diff > 0 and trans_diff > 0:
            both_better = 1
        if rot_diff < 0 and trans_diff < 0:
            both_better = -1
        
        if ours_rot_err < 10 and ours_trans_err < 10:
            ours_great = 1
        else:
            ours_great = 0

        # ours_great
        error_diff_tuple = (frame_pair_info, rot_diff.item(), ours_rot_err, vision_only_rot_err)


        error_diffs.append(error_diff_tuple)

    # TODO: extract top 10 samples based on rot error, trans error, and mix of both
    error_diffs.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    print("Top improvements (rotation, translation):")
    print(error_diffs[:20])
    print("Bottom improvements (rotation, translation):")
    print(error_diffs[-20:])   
    
    k = 100
    top_k_samples = error_diffs[:k]   
    
    bottom_k_samples = error_diffs[-k:]
    
    is_hm3d = True

    # for sample in top_k_samples + bottom_k_samples:
    #     frame_pair_info = sample[0]
    #     vid, source_frame_id, target_frame_id = frame_pair_info
    #     source_frame_image = os.path.join(frame_dir, vid, f"{source_frame_id:06d}.jpg")
    #     target_frame_image = os.path.join(frame_dir, vid, f"{target_frame_id:06d}.jpg")

    #     # show image pair in same plot
    #     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    #     source_image = plt.imread(source_frame_image)
    #     target_image = plt.imread(target_frame_image)
    #     axs[0].imshow(source_image)
    #    # axs[0].set_title(f"Source Frame: {source_frame_id}")
    #     axs[0].axis('off')
    #     axs[1].imshow(target_image)
    #     #axs[1].set_title(f"Target Frame: {target_frame_id}")       
    #     axs[1].axis('off')
    #     plt.tight_layout()
    #     if sample[1] == 1:
    #         best_string = "best"
    #     else:
    #         best_string = "worst"
        
    #     output_frames_dir = output_visualization_dir + "_" + best_string 
    #     os.makedirs(output_frames_dir, exist_ok=True)
    #     plt.savefig(f"{output_frames_dir}/{vid}_{source_frame_id}_{target_frame_id}.png")
    #     plt.close()


    
    # get audio clips
    # for sample in tqdm(top_k_samples + bottom_k_samples[::-1]):
    #for sample in [(("unc_basketball_03-16-23_01_26", 1515, 1545), 1)]:
    for sample in bottom_k_samples:
    # for sample in [(("uniandes_basketball_004_25", 1365, 1395 ), 1), (("minnesota_rockclimbing_030_28", 855, 885), 1)]:
        frame_pair_info = sample[0]
        if is_hm3d:
            source_label, target_label = frame_pair_info
            source_frame_image = frame_pairs_ours[frame_pair_info]["source_img_path"]
            target_frame_image = frame_pairs_ours[frame_pair_info]["target_img_path"]
            
            source_audio_file_path = frame_pairs_ours[frame_pair_info]["source_sound_path"]
            target_audio_file_path = frame_pairs_ours[frame_pair_info]["target_sound_path"]
            
            source_frame_id = source_label.split("-")[0]
            target_frame_id = target_label.split("-")[-1]
            vid = source_label.split("-")[1]
            
            rerr = sample[2]

        else:
            vid, source_frame_id, target_frame_id = frame_pair_info
            source_frame_image = os.path.join(frame_dir, vid, f"{source_frame_id:06d}.jpg")
            target_frame_image = os.path.join(frame_dir, vid, f"{target_frame_id:06d}.jpg")
      

            source_audio_file_path = os.path.join(audio_file_dir, vid, "sound", f"{source_frame_id:06d}_duration_1000ms.wav")
            target_audio_file_path = os.path.join(audio_file_dir, vid, "sound", f"{target_frame_id:06d}_duration_1000ms.wav")

            # source_doa_path = os.path.join(doa_dir, vid, f"{source_frame_id:06d}_duration_1000ms_azimuths.npy")
            # target_doa_path = os.path.join(doa_dir, vid, f"{target_frame_id:06d}_duration_1000ms_azimuths.npy")

            # if not os.path.exists(source_doa_path) or not os.path.exists(target_doa_path):
            #     raise Exception(f"DOA file not found! {source_doa_path}, {target_doa_path}")

            # source_doa = np.load(source_doa_path)
            # target_doa = np.load(target_doa_path)
            rerr = sample[2]

        # Save circular spectrum plots for source and target DOAs
        def _save_polar_doa(doa_vals, out_path, color='C0'):
            theta = np.deg2rad(np.arange(len(doa_vals)))
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='polar')
            ax.set_theta_zero_location('N')  # 0° up
            ax.set_theta_direction(1)        # CCW: 90° left, 180° down, 270° right
            ax.plot(theta, doa_vals, color=color, lw=2)
            ax.fill_between(theta, 0.0, doa_vals, color=color, alpha=0.25)
            max_val = float(np.max(doa_vals)) if doa_vals.size else 1.0
            ax.set_ylim(0, max(1e-6, max_val))
            ax.grid(True)
            fig.tight_layout(pad=0.5)
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)

        doa_output_dir = os.path.join(output_visualization_dir, "doa_plots")
        os.makedirs(doa_output_dir, exist_ok=True)

        # src_doa_img_path = os.path.join(doa_output_dir, f"{vid}_{source_frame_id:06d}_doa.png")
        # tgt_doa_img_path = os.path.join(doa_output_dir, f"{vid}_{target_frame_id:06d}_doa.png")
        # _save_polar_doa(source_doa, src_doa_img_path, color='C0')
        # _save_polar_doa(target_doa, tgt_doa_img_path, color='C0')
        

        source_audio_clip = AudioFileClip(source_audio_file_path)
        target_audio_clip = AudioFileClip(target_audio_file_path)

        H = 720
        left_img = ImageClip(source_frame_image).resize(height=H)
        right_img = ImageClip(target_frame_image).resize(height=H)

        panel_w = int(max(left_img.w, right_img.w))
        panel_h = int(max(left_img.h, right_img.h))
        left_img = left_img.on_color(size=(panel_w, panel_h), color=(0, 0, 0), pos=("center", "center"))
        right_img = right_img.on_color(size=(panel_w, panel_h), color=(0, 0, 0), pos=("center", "center"))

        active_color = (0, 255, 0)
        inactive_color = (40, 40, 40)
        mar = 12
        dim_factor = 0.75
        pause_duration = 1.0

        def with_active_border(c):
            return c.fx(vfx.margin, mar=mar, color=active_color)

        def with_inactive_border_dim(c):
            return c.fx(vfx.margin, mar=mar, color=inactive_color).fx(vfx.colorx, dim_factor)

        def with_neutral_border(c):
            return c.fx(vfx.margin, mar=mar, color=inactive_color)

        def _frame_id_str(x):
            try:
                return f"{int(x):06d}"
            except (TypeError, ValueError):
                return str(x)

        source_frame_id_str = _frame_id_str(source_frame_id)
        target_frame_id_str = _frame_id_str(target_frame_id)

        vision_only_rot_err = float(frame_pairs_vision_only[frame_pair_info]["rotation_error"])
        ours_rot_err = float(frame_pairs_ours[frame_pair_info]["rotation_error"])
        vision_only_trans_err = float(frame_pairs_vision_only[frame_pair_info]["translation_error"])
        ours_trans_err = float(frame_pairs_ours[frame_pair_info]["translation_error"])

        print(f"Processing video pair: {vid}, frames: {source_frame_id_str}, {target_frame_id_str}")
        print(f"Vision Only - Rot Err: {vision_only_rot_err:.2f}, Ours - Rot Err: {ours_rot_err:.2f}")
        print(f"Vision Only - Trans Err: {vision_only_trans_err:.2f}, Ours - Trans Err: {ours_trans_err:.2f}")

        best_string = "best" if sample[1] >= 0 else "worst"
        output_videos_dir = os.path.join(output_visualization_dir, f"videos_error_lines_{best_string}")
        full_vid_name = (
            f"{vid}_{source_frame_id_str}_{target_frame_id_str}"
            f"_vr{vision_only_rot_err:.2f}_or{ours_rot_err:.2f}"
            f"_vt{vision_only_trans_err:.2f}_ot{ours_trans_err:.2f}.mp4"
        )

        error_plot_dir = os.path.join(output_visualization_dir, "error_plots")
        os.makedirs(error_plot_dir, exist_ok=True)
        error_plot_path = os.path.join(
            error_plot_dir,
            f"{vid}_{source_frame_id_str}_{target_frame_id_str}_rotation_error.png",
        )

        def _save_error_plot(out_path, vision_err_deg, ours_err_deg):
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="polar")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([])
            ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
            ax.set_xticklabels([f"{d}°" for d in [0, 30, 60, 90, 120, 150, 180]])
            ax.grid(True, alpha=0.35)
            ax.set_title("Rotation Error in Degrees", pad=20)

            line_specs = [
                (0.0, 1.00, "k", "Ground truth: 0.0°", 3.5),
                (float(np.clip(vision_err_deg, 0.0, 180.0)), 0.82, "r", f"Vision-only: {vision_err_deg:.1f}°", 3.0),
                (float(np.clip(ours_err_deg, 0.0, 180.0)), 0.64, "g", f"Ours: {ours_err_deg:.1f}°", 3.0),
            ]

            for angle_deg, radius, color, label, lw in line_specs:
                theta = np.deg2rad([angle_deg, angle_deg])
                ax.plot(theta, [0.0, radius], color=color, lw=lw, label=label)
                ax.scatter([np.deg2rad(angle_deg)], [radius], color=color, s=55, zorder=3)

            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, -0.30),
                borderaxespad=1.5,
                frameon=False,
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        _save_error_plot(error_plot_path, vision_only_rot_err, ours_rot_err)

        def plot_clip(duration):
            clip = ImageClip(error_plot_path).resize(height=panel_h)
            if clip.w > panel_w:
                clip = clip.resize(width=panel_w)
            return clip.on_color(size=(panel_w, panel_h), color=(0, 0, 0), pos=("center", "center")).set_duration(duration)

        def build_row(duration, left_active=False, right_active=False):
            left_panel = (
                with_active_border(left_img).set_duration(duration)
                if left_active
                else with_inactive_border_dim(left_img).set_duration(duration)
            )
            right_panel = (
                with_active_border(right_img).set_duration(duration)
                if right_active
                else with_inactive_border_dim(right_img).set_duration(duration)
            )
            error_panel = with_neutral_border(plot_clip(duration))
            return clips_array([[left_panel, right_panel, error_panel]], bg_color=(0, 0, 0))

        pre_pause_seg = build_row(pause_duration)
        seg1_with_plot = build_row(source_audio_clip.duration, left_active=True).set_audio(source_audio_clip)
        pause_seg = build_row(pause_duration)
        seg2_with_plot = build_row(target_audio_clip.duration, right_active=True).set_audio(target_audio_clip)
        post_pause_seg = build_row(pause_duration)

        final_video = concatenate_videoclips(
            [pre_pause_seg, seg1_with_plot, pause_seg, seg2_with_plot, post_pause_seg],
            method="compose",
        )

        os.makedirs(output_videos_dir, exist_ok=True)
        final_video.write_videofile(
            os.path.join(output_videos_dir, full_vid_name),
            fps=1,
            threads=4,
        )

        final_video.close()
        source_audio_clip.close()
        target_audio_clip.close()

 
        # """
        # min_xyz = pts_all.min(axis=0)
        # max_xyz = pts_all.max(axis=0)
        # center = 0.5 * (min_xyz + max_xyz)
        # extent = max(max_xyz - min_xyz)  # largest side
        # margin = 0.07 * (extent if extent > 1e-6 else 1.0)  # smaller margin
        # half_span = 0.5 * extent + margin
        # MIN_RADIUS = 0.4  # allow tighter zoom
        # R = max(half_span, MIN_RADIUS)

        # ax.set_xlim(center[0] - R, center[0] + R)
        # ax.set_ylim(center[1] - R, center[1] + R)
        # ax.set_zlim(center[2] - R, center[2] + R)
        # ax.set_box_aspect((1, 1, 1))
        # ax.view_init(elev=25, azim=120)

        # # Legend
        # ax.plot([], [], [], color='k', lw=3.4, ls='-', label='Source')
        # ax.plot([], [], [], color='k', lw=2.8, ls='--', label='Target')
        # ax.plot([], [], [], color='r', lw=2.8, ls='-', label='Vision-only')
        # ax.plot([], [], [], color='g', lw=2.8, ls='-', label='Ours')
        # ax.legend(loc='upper left')

        # fig.tight_layout()
        # fig.savefig(pose_plot_path, dpi=300, bbox_inches='tight')
        # plt.close(fig)

        # def right_pose_panel(target_height, duration):
        #     # Keep same height as the left composite; width comes from the (now larger) figure
        #     return ImageClip(pose_plot_path).resize(height=target_height).set_duration(duration)

        # pre_pause_seg = clips_array(
        #     [
        #     [with_inactive_border_dim(left_img).set_duration(pause_duration),
        #      with_inactive_border_dim(right_img).set_duration(pause_duration)],
        #     [with_inactive_border_dim(left_doa_img).set_duration(pause_duration),
        #      with_inactive_border_dim(right_doa_img).set_duration(pause_duration)],
        #     ],
        #     bg_color=(0, 0, 0),
        # )

        # pause_seg = clips_array(
        #     [
        #     [with_inactive_border_dim(left_img).set_duration(pause_duration),
        #      with_inactive_border_dim(right_img).set_duration(pause_duration)],
        #     [with_inactive_border_dim(left_doa_img).set_duration(pause_duration),
        #      with_inactive_border_dim(right_doa_img).set_duration(pause_duration)],
        #     ],
        #     bg_color=(0, 0, 0),
        # )

        # seg2_dur = target_audio_clip.duration
        # seg2 = clips_array(
        #     [
        #     [with_inactive_border_dim(left_img).set_duration(seg2_dur),
        #      with_active_border(right_img).set_duration(seg2_dur)],
        #     [with_inactive_border_dim(left_doa_img).set_duration(seg2_dur),
        #      with_active_border(right_doa_img).set_duration(seg2_dur)],
        #     ],
        #     bg_color=(0, 0, 0),
        # ).set_audio(target_audio_clip)

        # seg1_with_pose = clips_array([[seg1, right_pose_panel(seg1.h, seg1_dur)]], bg_color=(0, 0, 0)).set_audio(seg1.audio)
        # pre_pause_with_pose = clips_array([[pre_pause_seg, right_pose_panel(pre_pause_seg.h, pause_duration)]], bg_color=(0, 0, 0)).set_audio(pre_pause_seg.audio)
        # pause_with_pose = clips_array([[pause_seg, right_pose_panel(pause_seg.h, pause_duration)]], bg_color=(0, 0, 0)).set_audio(pause_seg.audio)
        # seg2_with_pose = clips_array([[seg2, right_pose_panel(seg2.h, seg2_dur)]], bg_color=(0, 0, 0)).set_audio(seg2.audio)

        # post_pause_seg = clips_array(
        #     [
        #     [with_inactive_border_dim(left_img).set_duration(pause_duration),
        #      with_inactive_border_dim(right_img).set_duration(pause_duration)],
        #     [with_inactive_border_dim(left_doa_img).set_duration(pause_duration),
        #      with_inactive_border_dim(right_doa_img).set_duration(pause_duration)],
        #     ],
        #     bg_color=(0, 0, 0),
        # )
        # post_pause_with_pose = clips_array([[post_pause_seg, right_pose_panel(post_pause_seg.h, pause_duration)]], bg_color=(0, 0, 0)).set_audio(post_pause_seg.audio)

        # final_video = concatenate_videoclips(
        #     [pre_pause_with_pose, seg1_with_pose, pause_with_pose, seg2_with_pose, post_pause_with_pose],
        #     method="compose"
        # )

        # os.makedirs(output_videos_dir, exist_ok=True)
        # final_video.write_videofile(
        #     os.path.join(output_videos_dir, full_vid_name),
        #     fps=1,
        #     threads=4,
        # )
        # """