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

from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

from vine_prune.utils.io import read_pickle
from vine_prune.utils.paths import ASSET_DIR

def format_int(ind):
    return "{:0>6d}".format(ind)

data_dirs = [
            '/home/hfreeman/harry_ws/repos/pruner_track/datasets/rss_2026/DEMOS/chili_place_diverse/demos',
            "/home/hfreeman/harry_ws/repos/pruner_track/datasets/rss_2026/DEMOS/chili_place_diverse_0/demos",
            "/home/hfreeman/harry_ws/repos/pruner_track/datasets/rss_2026/DEMOS/chili_place_diverse_1/demos",
            "/home/hfreeman/harry_ws/repos/pruner_track/datasets/rss_2026/DEMOS/chili_place_diverse_2/demos",
            "/home/hfreeman/harry_ws/repos/pruner_track/datasets/rss_2026/DEMOS/chili_place_diverse_3/demos",
             ]
skip_exps = ['bad', 'failed']

subdirs = []
orig_exp_names = []
for data_dir in data_dirs:
        for scene_name in os.listdir(data_dir):
            scene_dir = os.path.join(data_dir, scene_name)
            for subdir_name in os.listdir(scene_dir):
                if subdir_name in skip_exps:
                    continue
            
                subdir = os.path.join(scene_dir, subdir_name)
                vid_path = os.path.join(subdir, 'training_data', 'vid.mp4')
                if not os.path.exists(vid_path):
                    print('no vid for: ', subdir_name)
                    continue

                subdirs.append(subdir)
                orig_exp_names.append(subdir)

                aug_dir = os.path.join(subdir, 'augmentations')
                if os.path.exists(aug_dir):
                    num_augs = 0
                    for aug_name in os.listdir(aug_dir):
                        vid_path = os.path.join(aug_dir, aug_name, 'training_data', 'vid.mp4')
                        if not os.path.exists(vid_path):
                            continue
                        num_augs += 1
                        
                        subdirs.append(os.path.join(aug_dir, aug_name))
                    #print(num_augs)

subdirs = sorted(subdirs)
orig_exp_names = sorted(orig_exp_names)

breakpoint()

output = '/home/hfreeman/harry_ws/repos/pruner_track/submodules/universal_manipulation_interface/example_demo_session/rss_2026_chili_place_plate_diverse.zarr.zip'

if True:

    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())

    all_videos = set()
    vid_args = list()

    buffer_start = 0

    for subdir in subdirs:
        training_data_dir = os.path.join(subdir, 'training_data')

        metadata_path = f'{training_data_dir}/metadata.pkl'
        metadata = read_pickle(metadata_path)

        gripper = metadata['gripper_info']
        gripper_id = 0

        episode_data = dict()

        eef_pose = gripper['tcp_pose']
        eef_pos = eef_pose[...,:3]
        eef_rot = eef_pose[...,3:]
        # gripper_widths = gripper['gripper_widths']

        # is_closed_orig = gripper['is_closed']
        # is_closed = np.zeros_like(is_closed_orig)
        # # TODO assuming one object grasp
        # closed_start = np.argwhere(is_closed_orig > 0).min()
        # closed_end = np.argwhere(is_closed_orig == 1.0).max() + 1
        # # TODO not sure if should round like this or set values normally
        # is_closed[closed_start:closed_end] = 1.0

        is_closed = gripper['is_closed']
        #first_closed = np.argwhere(is_closed).min()
        #is_closed[first_closed] = False
        is_closed = is_closed.astype(float)
        # TODO I AM CORRECTING THIS HERE BUT THIS SHOULD BE DONE EARLIER

        assert is_closed[0] == 0
        assert np.max(is_closed) > 0
        # assert is_closed[-1] == 0

        # set to match umi
        # gripper_widths[gripper_widths > 0.85] = 0.85
        demo_start_pose = np.empty_like(eef_pose)
        demo_start_pose[:] = gripper['demo_start_pose']
        demo_end_pose = np.empty_like(eef_pose)
        demo_end_pose[:] = gripper['demo_end_pose']

        robot_name = f'robot{gripper_id}'
        episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
        episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
        episode_data[robot_name + '_gripper_closed'] = np.expand_dims(is_closed, axis=-1).astype(np.float32)
        episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
        episode_data[robot_name + '_demo_end_pose'] = demo_end_pose

        out_replay_buffer.add_episode(data=episode_data, compressors=None)
    
        vid_info = metadata['vid_info']

        video_path = os.path.join(training_data_dir, 'vid.mp4')
        video_start, video_end = vid_info['video_start_end']

        n_frames = video_end - video_start

        cam_id = 0

        videos_dict = defaultdict(list)

        videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })

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

    valid_mask = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'valid_area.png'), -1)
    gripper_seg_mask = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'gripper_seg_mask.png'), -1)

    valid_mask = cv2.resize(valid_mask, (480, 360), interpolation=cv2.INTER_NEAREST)
    gripper_seg_mask = cv2.resize(gripper_seg_mask, (480, 360), interpolation=cv2.INTER_NEAREST)


    def video_to_zarr(replay_buffer, mp4_path, tasks):
        resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
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

                    img[gripper_seg_mask > 0] = 255
                    img[valid_mask == 0] = 0
                        
                    # mask out gripper
                    # img = draw_predefined_mask(img, color=(0,0,0), 
                    #     mirror=no_mirror, gripper=True, finger=False)

                    img = resize_tf(img)

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


    
