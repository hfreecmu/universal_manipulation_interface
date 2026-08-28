import os
import torch
import dill
import hydra
from urdfpy import URDF
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from multiprocessing.managers import SharedMemoryManager
import time
import math
import scipy.spatial.transform as st

from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.multi_uvc_camera import MultiUvcCamera
from umi.real_world.video_recorder import VideoRecorder
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from umi.common.precise_sleep import precise_wait

import gymnasium
import xarm_env

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from diffusion_policy.common.pytorch_util import dict_apply

from vine_prune.utils.general_utils import (
    create_pose,
)
from vine_prune.utils.paths import ASSET_DIR

VALID_MASK = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'valid_area.png'), -1)
GRIPPER_SEG_MASK = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'gripper_seg_mask.png'), -1)
mask_input_res = (960, 720)

def solve_table_collision(ee_pose, gripper_width, height_threshold, tcp_offset):
    if height_threshold is None:
        return
    
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * finger_thickness / 2, dy * gripper_width / 2, tcp_offset))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_workspace_collision(ee_pose, x_threshold, z_threshold, z_min_threshold):
    
    if x_threshold is not None:
        delta = min(x_threshold - ee_pose[0], 0)
        ee_pose[0] += delta

    if z_threshold is not None:
        delta = min(z_threshold - ee_pose[2], 0)
        ee_pose[2] += delta

    if z_min_threshold is not None:
          delta = max(z_min_threshold - ee_pose[2], 0)
          ee_pose[2] += delta

def get_obs(
        camera_obs_horizon,
        camera_down_sample_steps,
        frequency,
        camera,
        robot_env_unwarpped,
        dt,
        robot_obs_horizon,
        robot_down_sample_steps,
        gripper_obs_horizon,
        gripper_down_sample_steps,
        last_camera_data,
):
    k = math.ceil(
                camera_obs_horizon * camera_down_sample_steps \
                * (60 / frequency)) + 2 # they say 2 here is optional
    last_camera_data = camera.get(k=k, out=last_camera_data)

    last_robots_data = list()

    robot_obs = robot_env_unwarpped._get_obs()
    last_robots_data.append(robot_obs)
    ###
    
    ### align camera obs
    # only one camera
    camera_data = last_camera_data[0]
    last_timestamp = camera_data['timestamp'][-1]

    # align camera obs timestamps
    camera_obs_timestamps = last_timestamp - (
        np.arange(camera_obs_horizon)[::-1] * camera_down_sample_steps * dt)

    this_timestamps = camera_data['timestamp']
    this_idxs = list()
    for t in camera_obs_timestamps:
        nn_idx = np.argmin(np.abs(this_timestamps - t))
        this_idxs.append(nn_idx)
    
    camera_obs = dict()
    camera_obs[f'camera{0}_rgb'] = camera_data['color'][this_idxs]

    obs_data = dict(camera_obs)
    obs_data['timestamp'] = camera_obs_timestamps
    ###

    ### align robot obs
    robot_obs_timestamps = last_timestamp - (
        np.arange(robot_obs_horizon)[::-1] * robot_down_sample_steps * dt)
    
    last_robot_data = last_robots_data[0]
    
    robot_pose_interpolator = PoseInterpolator(
        t=last_robot_data['robot_timestamp'], 
        x=last_robot_data['ee_pose'])
    robot_pose = robot_pose_interpolator(robot_obs_timestamps)

    robot_obs = {
        f'robot{0}_eef_pos': robot_pose[...,:3],
        f'robot{0}_eef_rot_axis_angle': robot_pose[...,3:]
    }
    obs_data.update(robot_obs)
    ###

    ### align gripper obs
    gripper_obs_timestamps = last_timestamp - (
        np.arange(gripper_obs_horizon)[::-1] * gripper_down_sample_steps * dt)
    
    gripper_interpolator = get_interp1d(
                t=last_robot_data['gripper_timestamp'],
                x=last_robot_data['ee_gripper'][...,None]
            )
    
    # not sure if need to round or not
    gripper_obs = {
            f'robot{0}_gripper_closed': gripper_interpolator(gripper_obs_timestamps)
        }

    obs_data.update(gripper_obs)

    return obs_data, last_camera_data

# def run(robot_env):
def run(robot_env):
    with SharedMemoryManager() as shm_manager:
        ckpt_path = '/home/hfreeman/Downloads/epoch=0119-train_loss=0.011.ckpt'
        CAMERA_OBS_HORIZON = 2
        thresh_closed = True
        TRIAL_NUM=24

        STOP_WHEN_OPEN = True
        CLOSED_THRESH = 0.1
        OPEN_THRESH = 0.1

        HEIGHT_THRESHOLD = None#-0.04
        TCP_OFFSET = 0.164

        X_THRESHOLD = None #0.4 #None
        Z_THRESHOLD = None #0.2 #None
        Z_MIN_THRESHOLD = 0.05

        VIDEO_PATHS = [
                f'/home/hfreeman/Downloads/microwave_trial/{TRIAL_NUM}.mp4'
            ]
        
        for vp in VIDEO_PATHS:
            if os.path.exists(vp):
                raise RuntimeError('vp exists: ', vp)


        # payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        # cfg = payload['cfg']
        # print("model_name:", cfg.policy.obs_encoder.model_name)
        # print("dataset_path:", cfg.task.dataset.dataset_path)

        # cls = hydra.utils.get_class(cfg._target_)
        # workspace = cls(cfg)
        # workspace: BaseWorkspace
        # workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # device = torch.device('cuda')

        # policy = workspace.model
        # if cfg.training.use_ema:
        #     policy = workspace.ema_model
        # policy.eval().to(device)

        # policy.num_inference_steps = 16 # DDIM inference iterations
        # obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
        # action_pose_repr = cfg.task.pose_repr.action_pose_repr
        # print('obs_pose_rep', obs_pose_rep)
        # print('action_pose_repr', action_pose_repr)
        ###

        ### camera stuff
        reset_all_elgato_devices()
        time.sleep(0.5)
        v4l_paths = get_sorted_v4l_paths()
        rw, rh, col, row = optimal_row_cols(
                    n_cameras=len(v4l_paths),
                    in_wh_ratio=4/3,
                    max_resolution=(960, 960)
                )
        
        fps = 60
        capture_fps = [fps]
        cap_buffer_size = [1]
        res = (1920, 1080)
        resolution = [res]

        video_recorder = [
        VideoRecorder.create_hevc_nvenc(
            fps=f,
            input_pix_fmt='bgr24',
            bit_rate=3000*1000
        ) for f in capture_fps]

        def vis_tf(data, input_res=res):
            img = data['color']
            f = get_image_transform(
                input_res=input_res,
                output_res=(rw,rh),
                bgr_to_rgb=False
            )
            img = f(img)

            f_mask = get_image_transform(
                input_res=mask_input_res,
                output_res=(rw,rh), 
                is_mask=True)
            valid_mask = np.ascontiguousarray(f_mask(VALID_MASK))
            gripper_seg_mask = np.ascontiguousarray(f_mask(GRIPPER_SEG_MASK))

            img[gripper_seg_mask > 0] = 255
            img[valid_mask == 0] = 0

            data['color'] = img
            return data
        vis_transform =[vis_tf]

        def tf(data, input_res=res):
            img = data['color']
            f = get_image_transform(
                input_res=input_res,
                output_res=(224, 224), 
                # obs output rgb
                bgr_to_rgb=True)
            img = np.ascontiguousarray(f(img))

            f_mask = get_image_transform(
                input_res=mask_input_res,
                output_res=(224,224), 
                is_mask=True)
            valid_mask = np.ascontiguousarray(f_mask(VALID_MASK))
            gripper_seg_mask = np.ascontiguousarray(f_mask(GRIPPER_SEG_MASK))

            img[gripper_seg_mask > 0] = 255
            img[valid_mask == 0] = 0

            data['color'] = img
            return data
        transform = [tf]

        max_obs_buffer_size = 60
        # camera_obs_latency = 0.125 #0.17
        # camera_obs_latency = 0
        camera_obs_latency = 0.17
        # camera_obs_latency = 0.20

        enable_multi_cam_vis = True

        camera_obs_horizon=CAMERA_OBS_HORIZON
        camera_down_sample_steps=1

        robot_obs_horizon=2
        robot_down_sample_steps=1
        gripper_obs_horizon=2
        gripper_down_sample_steps=1

        frequency = 5
        # frequency = 10

        robot_obs_latency = 0.0001
        robot_action_latency = 0.1
        gripper_obs_latency = 0.01
        gripper_action_latency = 0.1
        action_exec_latency = 0.01
        
        # steps_per_inference = 6
        # steps_per_inference = 8
        steps_per_inference = 12
        # steps_per_inference = 16

        # mug is 10 and 8
        # chili is 10 and 8
        # sugar is 10 and 8

        dt = 1 / frequency

        if robot_env is None:
            robot_env_data = {
                'ip': '192.168.1.211',

                "ee_controller_frequency": 200.0,
                "ee_log_freq": None,
                'ee_max_pos_speed': 2.0,
                'ee_max_rot_speed': 6.0,
                "ee_history_len": 200,

                "gripper_controller_frequency": 60.0,
                'gripper_log_freq': None,
                "gripper_history_len": 60,

                'receive_robot_latency': robot_obs_latency,
                'receive_gripper_latency': gripper_obs_latency,

                'compensate_latency': True,
                'robot_action_latency': robot_action_latency,
                'gripper_action_latency': gripper_action_latency,

                'thresh_closed': thresh_closed,
            }

            # robot_env = gymnasium.make('xarm_sim_env/Xarm-v0', env_data=robot_env_data)
            robot_env = gymnasium.make('xarm_servo_env/Xarm-v0', env_data=robot_env_data)

        camera = MultiUvcCamera(
                dev_video_paths=v4l_paths,
                shm_manager=shm_manager,
                resolution=resolution,
                capture_fps=capture_fps,
                put_downsample=False,
                get_max_k=max_obs_buffer_size,
                receive_latency=camera_obs_latency,
                cap_buffer_size=cap_buffer_size,
                transform=transform,
                vis_transform=vis_transform,
                video_recorder=video_recorder,
                verbose=False
            )
        
        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        camera.start(wait=False)
        if multi_cam_vis is not None:
            multi_cam_vis.start(wait=False)

        camera.start_wait()
        if multi_cam_vis is not None:
            multi_cam_vis.start_wait()

        # waits for camera data
        time.sleep(2.0)
        last_camera_data = None
        ###
        
        _, _ = robot_env.reset()
        robot_env_unwarpped = robot_env.unwrapped

        # get current pose
        obs, last_camera_data = get_obs(
            camera_obs_horizon=camera_obs_horizon,
            camera_down_sample_steps=camera_down_sample_steps,
            frequency=frequency,
            camera=camera,
            robot_env_unwarpped=robot_env_unwarpped,
            dt=dt,
            robot_obs_horizon=robot_obs_horizon,
            robot_down_sample_steps=robot_down_sample_steps,
            gripper_obs_horizon=gripper_obs_horizon,
            gripper_down_sample_steps=gripper_down_sample_steps,
            last_camera_data=last_camera_data,
        )

        is_closed = obs['robot0_gripper_closed'][0]
        is_closed_prev = obs['robot0_gripper_closed'][0]

        payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']
        print("model_name:", cfg.policy.obs_encoder.model_name)
        print("dataset_path:", cfg.task.dataset.dataset_path)

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        device = torch.device('cuda')

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.eval().to(device)

        policy.num_inference_steps = 16 # DDIM inference iterations
        obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
        action_pose_repr = cfg.task.pose_repr.action_pose_repr
        print('obs_pose_rep', obs_pose_rep)
        print('action_pose_repr', action_pose_repr)

        episode_start_pose = None
        # if True:
        #     episode_start_pose = list()
        #     for robot_id in range(1):
        #         pose = np.concatenate([
        #             obs[f'robot{robot_id}_eef_pos'],
        #             obs[f'robot{robot_id}_eef_rot_axis_angle']
        #         ], axis=-1)[-1]
        #         episode_start_pose.append(pose)

        # warm up policy
        with torch.no_grad():
            policy.reset()
            obs_dict_np = get_real_umi_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta, 
                        obs_pose_repr=obs_pose_rep,
                        tx_robot1_robot0=np.eye(4),
                        episode_start_pose=episode_start_pose)
            episode_start_pose = None

            obs_dict = dict_apply(obs_dict_np, 
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            result = policy.predict_action(obs_dict)
            action = result['action_pred'][0].detach().to('cpu').numpy()
            action = get_real_umi_action(action, obs, action_pose_repr)
            del result

        test = None
        while test != 'c':
            test = input('Ready? ')
            if test == 'x':
                return

        # start episode
        policy.reset()
        # start_delay = 1.0
        start_delay = 2.0
        eval_t_start = time.time() + start_delay
        t_start = time.monotonic() + start_delay

        camera.restart_put(start_time=eval_t_start)
        camera.start_recording(video_path=VIDEO_PATHS, start_time=eval_t_start)

        frame_latency = 1/60
        precise_wait(eval_t_start - frame_latency, time_func=time.time)
        print("Started!")
        iter_idx=0

        try:
            while True:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                # get obs
                obs, last_camera_data = get_obs(
                    camera_obs_horizon=camera_obs_horizon,
                    camera_down_sample_steps=camera_down_sample_steps,
                    frequency=frequency,
                    camera=camera,
                    robot_env_unwarpped=robot_env_unwarpped,
                    dt=dt,
                    robot_obs_horizon=robot_obs_horizon,
                    robot_down_sample_steps=robot_down_sample_steps,
                    gripper_obs_horizon=gripper_obs_horizon,
                    gripper_down_sample_steps=gripper_down_sample_steps,
                    last_camera_data=last_camera_data,
                )

                is_prev_closed = is_closed
                is_closed = obs['robot0_gripper_closed'][0]
                if STOP_WHEN_OPEN:
                    if is_prev_closed > CLOSED_THRESH and is_closed < OPEN_THRESH:
                        print('Opened and stopping')
                        break

                obs_timestamps = obs['timestamp']
                print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                # if episode_start_pose is None:
                #     episode_start_pose = list()
                #     for robot_id in range(1):
                #         pose = np.concatenate([
                #             obs[f'robot{robot_id}_eef_pos'],
                #             obs[f'robot{robot_id}_eef_rot_axis_angle']
                #         ], axis=-1)[-1]
                #         episode_start_pose.append(pose)

                # run inference
                with torch.no_grad():
                    s = time.time()
                    obs_dict_np = get_real_umi_obs_dict(
                            env_obs=obs, shape_meta=cfg.task.shape_meta, 
                            obs_pose_repr=obs_pose_rep,
                            tx_robot1_robot0=np.eye(4),
                            episode_start_pose=episode_start_pose)

                    obs_dict = dict_apply(obs_dict_np, 
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                    result = policy.predict_action(obs_dict)
                    raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                    action = get_real_umi_action(raw_action, obs, action_pose_repr)
                    print('Inference latency:', time.time() - s)

                this_target_poses = action

                for target_pose in this_target_poses:
                    for robot_idx in range(1):
                        solve_table_collision(
                            ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                            gripper_width=0.84*(1 - np.round(np.clip(target_pose[robot_idx * 7 + 6], 0, 1))),
                            height_threshold=HEIGHT_THRESHOLD,
                            tcp_offset=TCP_OFFSET,
                        )

                        solve_workspace_collision(
                            ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                            x_threshold=X_THRESHOLD,
                            z_threshold=Z_THRESHOLD,
                            z_min_threshold=Z_MIN_THRESHOLD,
                        )

                # for target_pose in this_target_poses:
                #     for robot_idx in range(1):
                #         breakpoint()
                #         target_pose[robot_idx * 7: robot_idx * 7 + 6]
                        
                #         target_pose[robot_idx * 7 + 2] = max(target_pose[robot_idx * 7 + 2])


                #         solve_table_collision(
                #             ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                #             gripper_width=0.84*(1 - np.round(np.clip(target_pose[robot_idx * 7 + 6], 0, 1))),
                #             height_threshold=HEIGHT_THRESHOLD,
                #             tcp_offset=TCP_OFFSET,
                #         )

                curr_time = time.time()
                action_timestamps = (np.arange(len(action), dtype=np.float64)
                                ) * dt + obs_timestamps[-1]
                # action_timestamps = (np.arange(len(action), dtype=np.float64)
                #                 ) * dt + curr_time
                is_new = action_timestamps > (curr_time + action_exec_latency)
                # is_new[:] = True

                if np.sum(is_new) == 0:
                    # exceeded time budget, still do something
                    this_target_poses = this_target_poses[[-1]]
                    # schedule on next available step
                    next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                    action_timestamp = eval_t_start + (next_step_idx) * dt
                    print('Over budget', action_timestamp - curr_time)
                    action_timestamps = np.array([action_timestamp])
                else:
                    this_target_poses = this_target_poses[is_new]
                    action_timestamps = action_timestamps[is_new]

                ee_pose = this_target_poses[:, 0:6]
                ee_gripper = this_target_poses[:, 6]
                # print('ee_gripper', ee_gripper)
                robot_action = {
                    'ee_pose': ee_pose,
                    'ee_gripper': ee_gripper,
                    'timestamps': action_timestamps,
                }

                robot_env_unwarpped.step(action=robot_action)


                precise_wait(t_cycle_end - frame_latency)
                iter_idx += steps_per_inference
        except KeyboardInterrupt:
            print("Interrupted")
            robot_env_unwarpped.close()

        camera.stop_recording()

        if multi_cam_vis is not None:
            multi_cam_vis.stop(wait=False)
        camera.stop(wait=False)

        camera.stop_wait()
        if multi_cam_vis is not None:
            multi_cam_vis.stop_wait()


if __name__ == "__main__":
    # # robot env
    # robot_env_data = {
    #     'api': '192.168.1.212'
    # }
    # robot_env = gymnasium.make('xarm_env/Xarm-v0', env_data=robot_env_data)
    # run(robot_env)

    run(None)