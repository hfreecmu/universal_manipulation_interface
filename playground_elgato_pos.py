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

from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.multi_uvc_camera import MultiUvcCamera
from umi.real_world.video_recorder import VideoRecorder
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer

import gymnasium
import xarm_env

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from diffusion_policy.common.pytorch_util import dict_apply

from vine_prune.utils.general_utils import (
    create_pose, format_int,
)
from vine_prune.utils.paths import ASSET_DIR

VALID_MASK = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'valid_area.png'), -1)
GRIPPER_SEG_MASK = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'gripper_seg_mask.png'), -1)
mask_input_res = (960, 720)

with SharedMemoryManager() as shm_manager:
    ckpt_path = '/home/hfreeman/Downloads/epoch=0119-train_loss=0.011.ckpt'
    thresh_closed = True

    VID_DIR = '/home/hfreeman/Downloads/vis_pos'
    image_ind = 0

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

    # max_obs_buffer_size = 60
    max_obs_buffer_size = 60
    #camera_obs_latency = 0.125 #0.17
    # camera_obs_latency = 0
    camera_obs_latency = 0.17

    enable_multi_cam_vis = True

    camera_obs_horizon=2
    camera_down_sample_steps=1
    frequency = 10
    # frequency = 60

    # VIDEO_PATHS = [
    #     '/home/hfreeman/Downloads/elgato_pos.mp4'
    # ]
    

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
    time.sleep(1.0)
    last_camera_data = None
    ###

    # robot env
    robot_env_data = {
        'api': '192.168.1.211',
        'thresh_closed': thresh_closed,
    }
    robot_env = gymnasium.make('xarm_pos_env/Xarm-v0', env_data=robot_env_data)
    robot_obs, _ = robot_env.reset()
    #

    k = math.ceil(
                camera_obs_horizon * camera_down_sample_steps \
                * (60 / frequency)) + 2 # they say 2 here is optional
    last_camera_data = camera.get(k=k, out=last_camera_data)
    camera_data = last_camera_data[0]

    obs_image = camera_data['color'][-1]

    obs_vis = cv2.cvtColor(obs_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('obv_vis', obs_vis)
    cv2.imwrite(os.path.join(VID_DIR, f'{format_int(image_ind)}.jpg'),
                obs_vis)
    image_ind += 1
    cv2.waitKey(0)

    # start_time = time.time()
    # camera.restart_put(start_time)
    # camera.start_recording(video_path=VIDEO_PATHS, start_time=start_time)

    obs = dict(robot_obs)
    obs['rgb'] = obs_image

    prev_obs = obs

    episode_start_pose = None
    with torch.no_grad():
        policy.reset()

    try:
        # for ind in range(100):
        while True:
            rgb = np.stack([prev_obs['rgb'], obs['rgb']])

            camera_obs = dict()
            camera_obs[f'camera{0}_rgb'] = rgb
            
            obs_data = dict(camera_obs)

            eef_pos = np.stack([prev_obs['ee_pose'][0:3],
                                obs['ee_pose'][0:3]])
            eef_aa = np.stack([prev_obs['ee_pose'][3:6],
                                obs['ee_pose'][3:6]])
            robot_eef_obs = {
                    f'robot{0}_eef_pos': eef_pos,
                    f'robot{0}_eef_rot_axis_angle': eef_aa
                }
            obs_data.update(robot_eef_obs)

            eef_is_closed = np.array([prev_obs['ee_gripper'], obs['ee_gripper']])[:, None]
            gripper_obs = {
                    f'robot{0}_gripper_closed': eef_is_closed
                }
            obs_data.update(gripper_obs)

            # if episode_start_pose is None:
            #     episode_start_pose = list()
            #     for robot_id in range(1):
            #         pose = np.concatenate([
            #             obs_data[f'robot{robot_id}_eef_pos'],
            #             obs_data[f'robot{robot_id}_eef_rot_axis_angle']
            #         ], axis=-1)[-1]
            #         episode_start_pose.append(pose)

            # first one might be slower
            with torch.no_grad():
                obs_dict_np = get_real_umi_obs_dict(
                        env_obs=obs_data, shape_meta=cfg.task.shape_meta, 
                        obs_pose_repr=obs_pose_rep,
                        tx_robot1_robot0=np.eye(4),
                        episode_start_pose=episode_start_pose)

                obs_dict = dict_apply(obs_dict_np, 
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                result = policy.predict_action(obs_dict)
                raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                action = get_real_umi_action(raw_action, obs_data, action_pose_repr)
                del result
            
            # time.sleep(1)
            breakpoint()
            for action_ind in range(0, 9):
                # if action_ind % 3 != 0:
                #     continue
            # breakpoint()
            # for action_ind in range(1, 6):
                prev_obs = obs

                aa = action[action_ind]

                gripper_action = np.clip(aa[-1], 0, 1).round()
                # gripper_action = np.clip(aa[-1], 0, 1)

                robot_action = {
                    'ee_pose': aa[0:6],
                    'ee_gripper': gripper_action,
                }
                
                robot_obs = robot_env.step(action=robot_action)[0]

                k = math.ceil(
                    camera_obs_horizon * camera_down_sample_steps \
                    * (60 / frequency)) + 2 # they say 2 here is optional
                            
                last_camera_data = camera.get(k=k, out=last_camera_data)
                camera_data = last_camera_data[0]
                

                obs_image = camera_data['color'][-1]

                obs = dict(robot_obs)
                obs['rgb'] = obs_image

                obs_vis = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
                cv2.imshow('obv_vis', obs_vis)
                cv2.waitKey(1)

                cv2.imwrite(os.path.join(VID_DIR, f'{format_int(image_ind)}.jpg'),
                            obs_vis)
                image_ind += 1
        ###
    except KeyboardInterrupt:
        print("Interrupted")
        # camera.stop_recording()

    if multi_cam_vis is not None:
        multi_cam_vis.stop(wait=False)
    camera.stop(wait=False)

    camera.stop_wait()
    if multi_cam_vis is not None:
        multi_cam_vis.stop_wait()
