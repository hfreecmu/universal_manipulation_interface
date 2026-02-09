#!/usr/bin/env python3
"""
Build ONE combined ReplayBuffer zarr.zip from:
  (A) "real" demos (metadata.pkl + training_data/vid.mp4, plus augmentations)
  (B) teleop demos (metadata.json + frames/vid.mp4)

This version SLOWS DOWN human ("real") by HUMAN_SLOWDOWN by:
  - resampling states to a longer horizon (pos interp, rot SLERP, gripper interp+round)
  - repeating video frames to match that longer horizon

It also shows EVERY WRITTEN FRAME (so you will see repeated frames for slowed human).
"""

import os
import numpy as np
import cv2
import zarr
import av
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.transform import Rotation as R, Slerp

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

from umi.common.cv_util import get_image_transform as get_image_transform_real
from diffusion_policy.common.cv2_util import get_image_transform as get_image_transform_teleop

from vine_prune.utils.io import read_pickle, read_json
from vine_prune.utils.paths import ASSET_DIR


# -----------------------
# CONFIG (EDIT THESE)
# -----------------------
REAL_DATA_DIRS = ['/home/hfreeman/harry_ws/repos/pruner_track/datasets/rss_2026/DEMOS/sugar_rotate/demos',
            "/home/hfreeman/harry_ws/repos/pruner_track/datasets/rss_2026/DEMOS/sugar_rotate/demos/chili_place_plate_GX011741/extra_aug",
             ]
REAL_SKIP_EXPS = ['bad', 'double_aug_bak', 'extra_aug']

TELEOP_DATA_DIR = "/home/hfreeman/Downloads/teleop/sugar/success"
TELEOP_SKIP_EXPS = set()

OUTPUT_ZARR_ZIP = "/home/hfreeman/harry_ws/repos/pruner_track/submodules/universal_manipulation_interface/example_demo_session/combined_sugar_v2_slow.zarr.zip"

OUT_RES = (224, 224)          # (H,W)
CAM_ID = 0
NUM_WORKERS = 1
COMPRESSION_LEVEL = 99

# Teleop timing params
ROBOT_LATENCY = 0.0001
GRIPPER_LATENCY = 0.01
FPS = 60
MASK_INPUT_RES_TELEOP = (960, 720)  # (W,H)

NUM_SEL = 15

# Slowdown factor for REAL/HUMAN demos
HUMAN_SLOWDOWN = 5


# -----------------------
# Helpers
# -----------------------
def remove_duplicate_times(times, values_list):
    times = np.asarray(times)
    _, idx = np.unique(times, return_index=True)
    idx = np.sort(idx)
    res_values_list = []
    for values in values_list:
        res_values_list.append(values[idx])
    return times[idx], res_values_list


def list_real_subdirs(data_dirs, skip_exps, num_sel):
    subdir_dict = {}
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        for scene_name in os.listdir(data_dir):
            scene_dir = os.path.join(data_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            for subdir_name in os.listdir(scene_dir):
                if subdir_name in skip_exps:
                    continue
                subdir = os.path.join(scene_dir, subdir_name)
                vid_path = os.path.join(subdir, "training_data", "vid.mp4")
                if not os.path.exists(vid_path):
                    print("no vid for:", subdir)
                    continue
                subdir_dict[subdir] = [subdir]

                aug_dir = os.path.join(subdir, "augmentations")
                if os.path.isdir(aug_dir):
                    for aug_name in os.listdir(aug_dir):
                        aug_subdir = os.path.join(aug_dir, aug_name)
                        aug_vid = os.path.join(aug_subdir, "training_data", "vid.mp4")
                        if os.path.exists(aug_vid):
                            subdir_dict[subdir].append(aug_subdir)

    if num_sel is not None and len(subdir_dict) > 0:
        keys = list(subdir_dict.keys())
        sel_inds = np.random.choice(len(keys), min(num_sel, len(keys)), replace=False)
        subdir_dict = {keys[i]: subdir_dict[keys[i]] for i in sel_inds}

    subdirs = []
    for key in subdir_dict:
        subdirs += subdir_dict[key]

    return sorted(subdirs)


def list_teleop_subdirs(data_dir, skip_exps, num_sel):
    subdirs = []
    if not os.path.isdir(data_dir):
        return subdirs
    for subdir_name in os.listdir(data_dir):
        if subdir_name in skip_exps:
            continue
        subdir = os.path.join(data_dir, subdir_name)
        vid_path = os.path.join(subdir, "frames", "vid.mp4")
        if not os.path.exists(vid_path):
            print("no vid for:", subdir)
            continue
        subdirs.append(subdir)

    if num_sel is not None and len(subdirs) > 0:
        sel_inds = np.random.choice(len(subdirs), min(num_sel, len(subdirs)), replace=False)
        subdirs = [subdirs[i] for i in sel_inds]

    return sorted(subdirs)


def get_video_hw(mp4_path):
    with av.open(mp4_path) as container:
        st = container.streams.video[0]
        return st.height, st.width  # (H,W)


def load_masks_real():
    valid = cv2.imread(os.path.join(ASSET_DIR, "gripper_masks", "valid_area.png"), -1)
    seg = cv2.imread(os.path.join(ASSET_DIR, "gripper_masks", "gripper_seg_mask.png"), -1)
    valid = cv2.resize(valid, (480, 360), interpolation=cv2.INTER_NEAREST)
    seg = cv2.resize(seg, (480, 360), interpolation=cv2.INTER_NEAREST)
    return valid, seg


def load_masks_teleop():
    valid = cv2.imread(os.path.join(ASSET_DIR, "gripper_masks", "valid_area.png"), -1)
    seg = cv2.imread(os.path.join(ASSET_DIR, "gripper_masks", "gripper_seg_mask.png"), -1)
    return valid, seg


# -----------------------
# Core: append episodes + build image tasks
# -----------------------
def append_real_episode_and_tasks(out_rb, subdir, buffer_start):
    training_data_dir = os.path.join(subdir, "training_data")
    metadata_path = os.path.join(training_data_dir, "metadata.pkl")
    metadata = read_pickle(metadata_path)

    gripper = metadata["gripper_info"]
    gripper_id = 0

    vid_info = metadata["vid_info"]
    video_path = os.path.join(training_data_dir, "vid.mp4")
    video_start, video_end = vid_info["video_start_end"]
    T_old = int(video_end - video_start)

    # Original per-frame arrays for the segment we will use
    eef_pose = gripper["tcp_pose"][:T_old]
    eef_pos = eef_pose[..., :3].astype(np.float64)
    eef_rot = eef_pose[..., 3:].astype(np.float64)

    is_closed = gripper["is_closed"][:T_old].astype(np.float64)
    # keep your original assumptions
    assert is_closed[0] == 0
    assert np.max(is_closed) > 0

    # ---- resample state to longer horizon ----
    S = float(HUMAN_SLOWDOWN)
    T_new = int(np.round(T_old * S))
    T_new = max(1, T_new)

    t_old = np.arange(T_old, dtype=np.float64)
    t_new = np.linspace(0.0, T_old - 1.0, T_new, dtype=np.float64)

    eef_pos_new = np.zeros((T_new, 3), dtype=np.float32)
    for i in range(3):
        eef_pos_new[:, i] = np.interp(t_new, t_old, eef_pos[:, i]).astype(np.float32)

    rots_old = R.from_rotvec(eef_rot)
    slerp = Slerp(t_old, rots_old)
    eef_rot_new = slerp(t_new).as_rotvec().astype(np.float32)

    is_closed_new = np.interp(t_new, t_old, is_closed).astype(np.float32)
    is_closed_new = np.round(np.clip(is_closed_new, 0.0, 1.0)).astype(np.float32)

    eef_pose_new = np.concatenate([eef_pos_new, eef_rot_new], axis=-1)
    demo_start_pose = np.empty_like(eef_pose_new); demo_start_pose[:] = eef_pose_new[0]
    demo_end_pose   = np.empty_like(eef_pose_new); demo_end_pose[:]   = eef_pose_new[-1]

    robot_name = f"robot{gripper_id}"
    episode_data = {
        robot_name + "_eef_pos": eef_pos_new,
        robot_name + "_eef_rot_axis_angle": eef_rot_new,
        robot_name + "_gripper_closed": np.expand_dims(is_closed_new, axis=-1),
        robot_name + "_demo_start_pose": demo_start_pose,
        robot_name + "_demo_end_pose": demo_end_pose,
    }
    out_rb.add_episode(data=episode_data, compressors=None)

    tasks = [{
        "camera_idx": CAM_ID,
        "frame_start": int(video_start),
        "frame_end": int(video_end),
        "buffer_start": int(buffer_start),
        "kind": "real",
        "T_new": int(T_new),
    }]

    # IMPORTANT: advance by slowed length
    return video_path, tasks, buffer_start + T_new


def append_teleop_episode_and_tasks(out_rb, subdir, buffer_start):
    vid_path = os.path.join(subdir, "frames", "vid.mp4")
    metadata_path = os.path.join(subdir, "metadata.json")
    metadata = read_json(metadata_path)

    start_time = metadata[0]["start_time"]

    eef_pose = []
    is_closed = []
    robot_timestamps = []
    gripper_timestamps = []

    for md in metadata:
        eef_pose.append(md["ee_pose"])
        is_closed.append(md["ee_gripper"])
        robot_timestamps.append(md["robot_receive_timestamp"])
        gripper_timestamps.append(md["gripper_receive_timestamp"])

    eef_pose = np.array(eef_pose)
    is_closed = np.array(is_closed)
    robot_timestamps = np.array(robot_timestamps) - ROBOT_LATENCY - start_time
    gripper_timestamps = np.array(gripper_timestamps) - GRIPPER_LATENCY - start_time

    eef_pos = eef_pose[..., :3]
    eef_rot = eef_pose[..., 3:]

    robot_timestamps, (eef_pos, eef_rot) = remove_duplicate_times(robot_timestamps, [eef_pos, eef_rot])
    gripper_timestamps, [is_closed] = remove_duplicate_times(gripper_timestamps, [is_closed])

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {vid_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_frames <= 0:
        with av.open(vid_path) as c:
            st = c.streams.video[0]
            n_frames = int(st.frames)

    dt = 1 / FPS
    image_timestamps = (np.arange(n_frames) + 1) * dt

    eef_pos_interp = np.zeros((len(image_timestamps), 3))
    for i in range(3):
        eef_pos_interp[:, i] = np.interp(image_timestamps, robot_timestamps, eef_pos[:, i])

    rotations = R.from_rotvec(eef_rot)
    slerp = Slerp(robot_timestamps, rotations)
    eef_rot_interp = slerp(image_timestamps).as_rotvec()

    is_closed_interp = np.interp(image_timestamps, gripper_timestamps, is_closed)
    is_closed_interp = np.round(np.clip(is_closed_interp, 0.0, 1.0))

    eef_pos = eef_pos_interp
    eef_rot = eef_rot_interp
    eef_pose = np.concatenate((eef_pos, eef_rot), axis=-1)
    is_closed = is_closed_interp

    demo_start_pose = np.empty_like(eef_pose); demo_start_pose[:] = eef_pose[0]
    demo_end_pose   = np.empty_like(eef_pose); demo_end_pose[:]   = eef_pose[-1]

    robot_name = "robot0"
    episode_data = {
        robot_name + "_eef_pos": eef_pos.astype(np.float32),
        robot_name + "_eef_rot_axis_angle": eef_rot.astype(np.float32),
        robot_name + "_gripper_closed": np.expand_dims(is_closed, axis=-1).astype(np.float32),
        robot_name + "_demo_start_pose": demo_start_pose,
        robot_name + "_demo_end_pose": demo_end_pose,
    }
    out_rb.add_episode(data=episode_data, compressors=None)

    tasks = [{
        "camera_idx": CAM_ID,
        "frame_start": 0,
        "frame_end": int(n_frames),
        "buffer_start": int(buffer_start),
        "kind": "teleop",
    }]

    return vid_path, tasks, buffer_start + n_frames


# -----------------------
# Image dumping
# -----------------------
def make_camera_dataset(out_rb, out_res, compressor):
    name = f"camera{CAM_ID}_rgb"
    T = out_rb["robot0_eef_pos"].shape[0]
    out_rb.data.require_dataset(
        name=name,
        shape=(T,) + tuple(out_res) + (3,),
        chunks=(1,) + tuple(out_res) + (3,),
        compressor=compressor,
        dtype=np.uint8,
    )


def video_to_zarr_mixed(replay_buffer, mp4_path, tasks, out_res, real_masks, teleop_masks):
    tasks = sorted(tasks, key=lambda x: x["frame_start"])
    cam = tasks[0]["camera_idx"]
    name = f"camera{cam}_rgb"
    img_array = replay_buffer.data[name]

    ih, iw = get_video_hw(mp4_path)
    resize_tf_real = get_image_transform_real(in_res=(iw, ih), out_res=out_res)
    resize_tf_teleop = get_image_transform_teleop(input_res=(iw, ih), output_res=out_res)

    valid_real, seg_real = real_masks
    valid_tele, seg_tele = teleop_masks

    resize_mask_tf = get_image_transform_teleop(
        input_res=MASK_INPUT_RES_TELEOP,
        output_res=out_res,
        is_mask=True,
    )
    valid_tele_r = np.ascontiguousarray(resize_mask_tf(valid_tele))
    seg_tele_r = np.ascontiguousarray(resize_mask_tf(seg_tele))

    # per-task cached repeat schedule (only used for real)
    repeats = None

    curr_task_idx = 0
    with av.open(mp4_path) as container:
        in_stream = container.streams.video[0]
        in_stream.thread_count = 1

        buffer_idx = 0
        for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
            if curr_task_idx >= len(tasks):
                break

            t = tasks[curr_task_idx]
            if frame_idx < t["frame_start"]:
                continue
            if frame_idx >= t["frame_end"]:
                raise RuntimeError("Frame indexing logic error")

            # initialize per-task state at task start
            if frame_idx == t["frame_start"]:
                buffer_idx = t["buffer_start"]
                if t["kind"] == "real":
                    T_old = int(t["frame_end"] - t["frame_start"])
                    T_new = int(t["T_new"])
                    src = np.round(np.linspace(0, T_old - 1, T_new)).astype(np.int64)
                    src = np.clip(src, 0, T_old - 1)
                    repeats = np.bincount(src, minlength=T_old).astype(np.int64)
                else:
                    repeats = None

            img = frame.to_ndarray(format="rgb24")

            if t["kind"] == "real":
                try:
                    img[seg_real > 0] = 255
                    img[valid_real == 0] = 0
                except ValueError:
                    seg_r = cv2.resize(seg_real, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    val_r = cv2.resize(valid_real, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img[seg_r > 0] = 255
                    img[val_r == 0] = 0

                img = resize_tf_real(img)

                local_idx = frame_idx - t["frame_start"]
                k = int(repeats[local_idx])
                for _ in range(k):
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    cv2.imshow("test", img)
                    cv2.waitKey(1)

                if (frame_idx + 1) == t["frame_end"]:
                    curr_task_idx += 1

            elif t["kind"] == "teleop":
                img = resize_tf_teleop(img)
                img[seg_tele_r > 0] = 255
                img[valid_tele_r == 0] = 0

                img_array[buffer_idx] = img
                buffer_idx += 1
                cv2.imshow("test", img)
                cv2.waitKey(1)

                if (frame_idx + 1) == t["frame_end"]:
                    curr_task_idx += 1

            else:
                raise ValueError(f"Unknown kind: {t['kind']}")


# -----------------------
# Main
# -----------------------
def main():
    cv2.setNumThreads(1)

    real_subdirs = list_real_subdirs(REAL_DATA_DIRS, REAL_SKIP_EXPS, NUM_SEL)
    teleop_subdirs = list_teleop_subdirs(TELEOP_DATA_DIR, TELEOP_SKIP_EXPS, NUM_SEL)

    print("real subdirs:", len(real_subdirs))
    print("teleop subdirs:", len(teleop_subdirs))

    out_rb = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    vid_args = []
    all_videos = set()
    buffer_start = 0

    for subdir in real_subdirs:
        mp4_path, tasks, buffer_start = append_real_episode_and_tasks(out_rb, subdir, buffer_start)
        vid_args.append((str(mp4_path), tasks))
        all_videos.add(str(mp4_path))

    for subdir in teleop_subdirs:
        mp4_path, tasks, buffer_start = append_teleop_episode_and_tasks(out_rb, subdir, buffer_start)
        vid_args.append((str(mp4_path), tasks))
        all_videos.add(str(mp4_path))

    print(f"{len(all_videos)} videos used in total!")
    print("total frames in combined buffer:", out_rb["robot0_eef_pos"].shape[0])

    img_compressor = JpegXl(level=COMPRESSION_LEVEL, numthreads=1)
    make_camera_dataset(out_rb, OUT_RES, img_compressor)

    real_masks = load_masks_real()
    teleop_masks = load_masks_teleop()

    num_workers = NUM_WORKERS if NUM_WORKERS is not None else multiprocessing.cpu_count()

    with tqdm(total=len(vid_args)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    pbar.update(len(completed))

                futures.add(executor.submit(
                    video_to_zarr_mixed,
                    out_rb, mp4_path, tasks, OUT_RES, real_masks, teleop_masks
                ))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

        for f in completed:
            f.result()

    print(f"Saving ReplayBuffer to {OUTPUT_ZARR_ZIP}")
    with zarr.ZipStore(OUTPUT_ZARR_ZIP, mode="w") as zip_store:
        out_rb.save_to_store(store=zip_store)
    print("Done.")


if __name__ == "__main__":
    main()
