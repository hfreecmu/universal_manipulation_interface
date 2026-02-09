import os
import numpy as np
import torch
import cv2
import glob
import imageio.v2 as imageio
import zarr
import pickle
import av
from tqdm import tqdm
import concurrent.futures
import multiprocessing
from collections import defaultdict
from scipy.spatial.transform import Rotation as R, Slerp

# from umi.common.cv_util import (
#     parse_fisheye_intrinsics,
#     FisheyeRectConverter,
#     get_image_transform, 
#     draw_predefined_mask,
#     inpaint_tag,
#     get_mirror_crop_slices
# )

# because this is reading the 1920 x 1080 we are using this one
from diffusion_policy.common.cv2_util import (
    get_image_transform
)

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

from vine_prune.utils.io import read_json
from vine_prune.utils.paths import ASSET_DIR

def format_int(ind):
    return "{:0>6d}".format(ind)

def remove_duplicate_times(times, values_list):
    times = np.asarray(times)
    _, idx = np.unique(times, return_index=True)
    idx = np.sort(idx)

    res_values_list = []
    for values in values_list:
        res_values_list.append(values[idx])

    return times[idx], res_values_list

def read_video_to_numpy_cv2(video_path, convert_rgb=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.copy(frame)

        # OpenCV reads in BGR by default
        if convert_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from: {video_path}")

    return np.stack(frames, axis=0)  # (T, H, W, C)

ROBOT_LATENCY = 0.0001
GRIPPER_LATENCY = 0.01
FPS = 60

data_dir = '/home/hfreeman/Downloads/teleop/chili_place_plate/success'
skip_exps = []

subdirs = []
for subdir_name in os.listdir(data_dir):
    if subdir_name in skip_exps:
        continue

    subdir = os.path.join(data_dir, subdir_name)
    vid_path = os.path.join(subdir, 'frames', 'vid.mp4')
    if not os.path.exists(vid_path):
        print('no vid for: ', subdir_name)
        continue

    subdirs.append(subdir)

subdirs = sorted(subdirs)

output = '/home/hfreeman/harry_ws/repos/pruner_track/submodules/universal_manipulation_interface/example_demo_session/chili_teleop.zarr.zip'

if True:

    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())

    all_videos = set()
    vid_args = list()

    buffer_start = 0

    for subdir in subdirs:
        vid_path = os.path.join(subdir, 'frames', 'vid.mp4')

        metadata_path = f'{subdir}/metadata.json'
        metadata = read_json(metadata_path)

        # should be same for all
        start_time = metadata[0]['start_time']

        eef_pose = []
        is_closed = []

        robot_timestamps = []
        gripper_timestamps = []

        for md in metadata:
            ee_pose = md['ee_pose']
            ee_gripper = md['ee_gripper']

            robot_receive_timestamp = md['robot_receive_timestamp']
            gripper_receive_timestamp = md['gripper_receive_timestamp']

            eef_pose.append(ee_pose)
            is_closed.append(ee_gripper)
            robot_timestamps.append(robot_receive_timestamp)
            gripper_timestamps.append(gripper_receive_timestamp)

        eef_pose = np.array(eef_pose)
        is_closed = np.array(is_closed)
        robot_timestamps = np.array(robot_timestamps) - ROBOT_LATENCY - start_time
        gripper_timestamps = np.array(gripper_timestamps) - GRIPPER_LATENCY - start_time

        eef_pos = eef_pose[...,:3]
        eef_rot = eef_pose[...,3:]

        robot_timestamps_orig = np.copy(robot_timestamps)
        robot_timestamps, (eef_pos, eef_rot) = remove_duplicate_times(robot_timestamps_orig, [eef_pos, eef_rot])
        if robot_timestamps_orig.shape[0] != robot_timestamps.shape[0]:
            print('WARNINNG DUPLICATE ROBOT TS')

        gripper_timestamps_orig = np.copy(gripper_timestamps)
        gripper_timestamps, [is_closed] = remove_duplicate_times(gripper_timestamps_orig, [is_closed])
        if gripper_timestamps_orig.shape[0] != gripper_timestamps.shape[0]:
            print('WARNINNG DUPLICATE GRIPPER TS')

        # now we need to interpolate
        dt = 1/FPS
        images = read_video_to_numpy_cv2(vid_path)

        # the +1 is for the time the first frame comes in
        image_timestamps = (np.arange(images.shape[0]) + 1) * dt

        eef_pos_interp = np.zeros((len(image_timestamps), 3))
        for i in range(3):
            eef_pos_interp[:, i] = np.interp(
                image_timestamps,
                robot_timestamps,
                eef_pos[:, i]
            )

        rotations = R.from_rotvec(eef_rot)

        slerp = Slerp(robot_timestamps, rotations)

        rot_interp = slerp(image_timestamps)
        eef_rot_interp = rot_interp.as_rotvec()

        is_closed_interp = np.interp(
            image_timestamps,
            gripper_timestamps,
            is_closed
        )

        # we are rounding here
        is_closed_interp = np.round(np.clip(is_closed_interp, 0.0, 1.0))

        eef_pos = eef_pos_interp
        eef_rot = eef_rot_interp
        eef_pose = np.concatenate((eef_pos, eef_rot), axis=-1)
        is_closed = is_closed_interp

        demo_start_pose = np.empty_like(eef_pose)
        demo_start_pose[:] = eef_pose[0]
        demo_end_pose = np.empty_like(eef_pose)
        demo_end_pose[:] = eef_pose[-1]

        gripper_id = 0
        episode_data = dict()
        robot_name = f'robot{gripper_id}'
        episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
        episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
        episode_data[robot_name + '_gripper_closed'] = np.expand_dims(is_closed, axis=-1).astype(np.float32)
        episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
        episode_data[robot_name + '_demo_end_pose'] = demo_end_pose

        out_replay_buffer.add_episode(data=episode_data, compressors=None)

        cam_id = 0
        video_start = 0
        video_end = images.shape[0]
        videos_dict = defaultdict(list)
        videos_dict[str(vid_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })

        n_frames = images.shape[0]
        buffer_start += n_frames

        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")

    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
     # dump images
    compression_level=99
    img_compressor = JpegXl(level=compression_level, numthreads=1)

    out_res = '224,224'
    out_res = tuple(int(x) for x in out_res.split(','))

    for cam_id in range(1):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    VALID_MASK = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'valid_area.png'), -1)
    GRIPPER_SEG_MASK = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'gripper_seg_mask.png'), -1)
    mask_input_res = (960, 720)

    def video_to_zarr(replay_buffer, mp4_path, tasks):
        # resize_tf = get_image_transform(
        #     in_res=(iw, ih),
        #     out_res=out_res
        # )

        resize_tf = get_image_transform(
            input_res=(iw, ih),
            output_res=out_res,
        )

        resize_mask_tf = get_image_transform(
            input_res=mask_input_res,
            output_res=out_res,
            is_mask=True,
        )
        valid_mask = np.ascontiguousarray(resize_mask_tf(VALID_MASK))
        gripper_seg_mask = np.ascontiguousarray(resize_mask_tf(GRIPPER_SEG_MASK))

        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        curr_task_idx = 0
        
        is_mirror = None
        
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')

                    #img[gripper_seg_mask > 0] = 255
                    #img[valid_mask == 0] = 0
                        
                    # mask out gripper
                    # img = draw_predefined_mask(img, color=(0,0,0), 
                    #     mirror=no_mirror, gripper=True, finger=False)

                    img = resize_tf(img)

                    img[gripper_seg_mask > 0] = 255
                    img[valid_mask == 0] = 0

                    cv2.imshow('test', img)
                    cv2.waitKey(1)

                    # print(episode_data[robot_name + '_gripper_closed'][frame_idx])
                    # if episode_data[robot_name + '_gripper_closed'][frame_idx] == 1.0:
                    #     breakpoint()

                    # compress image
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False

    num_workers = 1
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)

    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")