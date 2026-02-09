import os
import torch
import dill
import hydra
from urdfpy import URDF
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

import gymnasium
import splat_env

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from diffusion_policy.common.pytorch_util import dict_apply

from vine_prune.utils.io import (
    read_np_data,
    read_pickle,
    read_yaml,
)
from vine_prune.utils.general_utils import (
    create_pose,
    read_config,
    read_cam_info,
)

from vine_prune.utils.paths import (
    GAUSSIAN_SPLATTING_DIR,
    ASSET_DIR,
    GOPRO_CALIB_DIR,
)

import sys
sys.path.append(GAUSSIAN_SPLATTING_DIR)
from scene.gaussian_model import GaussianModel

if True:
    ckpt_path = '/home/hfreeman/Downloads/epoch=0119-train_loss=0.010.ckpt'
    data_dir = "/home/hfreeman/harry_ws/repos/pruner_track/datasets/rss_2026/DEMOS/erase_plate/demos/GX010974/tmp/GX011535"
    AUGMENT_EXTR = True
    AUGMENT_INTR = False

    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)
    
    cfg['training']['seed'] = np.random.randint(0, 15000)

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

    ee_interp_data_path = os.path.join(data_dir, 'hand_obj', 'ee_interp_opt.npy')
    ee_interp_data = read_np_data(ee_interp_data_path)

    obj_data_list = ee_interp_data['objects']
    assert len(obj_data_list) == 1
    obj_data = obj_data_list[0]
    label_identifier = obj_data['label_identifier']

    ee_data = ee_interp_data['ee']

    drive_joint = ee_interp_data['drive_joints']
    is_closed = ee_interp_data['is_closed']

    obj_splat_path = os.path.join(data_dir, 'meshes', label_identifier, 'opt', 'object_splat_pca_opt.ply')
    obj_gaussians = GaussianModel(3)
    obj_gaussians.load_ply(obj_splat_path)

    scene_gaussians = GaussianModel(3)
    scene_splat_path = os.path.join(data_dir, 'splat', 'splat.ply')
    scene_gaussians.load_ply(scene_splat_path)

    ee_gaussians = GaussianModel(3)
    gripper_splat_path = os.path.join(ASSET_DIR, 'gripper_splat_post_opt.ply')
    ee_gaussians.load_ply(gripper_splat_path)

    gripper_segment_path = os.path.join(ASSET_DIR, 'segmented_gripper.pkl')
    segment_info = read_pickle(gripper_segment_path)
    seg_ind_dict = segment_info['label_dict']

    urdf_path = os.path.join(ASSET_DIR, 'gripper.urdf')
    robot = URDF.load(urdf_path)

    ee_2_cam_path = os.path.join(ASSET_DIR, 'ee_2_cam.yml')
    ee_2_cam = read_yaml(ee_2_cam_path)
    ee_2_cam_quat = np.array([
        ee_2_cam['transform']['rotation']['x'],
        ee_2_cam['transform']['rotation']['y'],
        ee_2_cam['transform']['rotation']['z'],
        ee_2_cam['transform']['rotation']['w'],
    ])
    ee_2_cam_rot = R.from_quat(ee_2_cam_quat).as_matrix()
    ee_2_cam_trans = np.array([
        ee_2_cam['transform']['translation']['x'],
        ee_2_cam['transform']['translation']['y'],
        ee_2_cam['transform']['translation']['z'],
    ])

    if AUGMENT_EXTR:
        # --- Rotation perturbation ---
        rot_jitter_deg = 5.0  # max ~5 degrees
        jitter_axis = np.random.randn(3)
        jitter_axis /= (np.linalg.norm(jitter_axis) + 1e-9)
        
        # Option 1: Uniform distribution in [-rot_jitter_deg, +rot_jitter_deg]
        jitter_angle = np.deg2rad(rot_jitter_deg) * (2 * np.random.rand() - 1)
        
        # Option 2: Normal distribution with rot_jitter_deg as std dev
        # jitter_angle = np.deg2rad(rot_jitter_deg) * np.random.randn()
        
        jitter_rot = R.from_rotvec(jitter_axis * jitter_angle).as_matrix()
        
        # Apply rotation jitter (in world frame)
        # ee_2_cam_rot = jitter_rot @ ee_2_cam_rot
        
        # --- Translation perturbation ---
        jitter_v = np.random.normal(size=3)
        jitter_v /= np.linalg.norm(jitter_v)
        mag = np.random.uniform(0.0, 0.015) 
        jitter_trans = mag * jitter_v
        
        # Apply translation jitter
        # ee_2_cam_trans = ee_2_cam_trans + jitter_trans

        M_jitter = create_pose(jitter_rot, jitter_trans)
    else:
        M_jitter = np.eye(4)

    M_ee2cam = create_pose(ee_2_cam_rot, ee_2_cam_trans)
    M_ee2cam = M_ee2cam @ M_jitter

    exp_info = read_config(data_dir)
    min_drive_joint = exp_info['min_drive_joint']

    fisheye_calib_dir = os.path.join(GOPRO_CALIB_DIR, exp_info['fisheye_calib'])
    fisheye_camera_info = read_cam_info(fisheye_calib_dir)
    fisheye_K = torch.FloatTensor(np.array(fisheye_camera_info['K']).reshape(3, 3)).cuda()
    fisheye_D = torch.FloatTensor(np.array(fisheye_camera_info['D'])).cuda()
    fisheye_dims = [fisheye_camera_info['height'], fisheye_camera_info['width']]

    if AUGMENT_INTR:
        percent = 0.01
        fisheye_K[0, 0] += torch.randn(1)[0].cuda() * percent * fisheye_K[0, 0]
        fisheye_K[1, 1] += torch.randn(1)[0].cuda() * percent * fisheye_K[1, 1]
        fisheye_K[0, 2] += torch.randn(1)[0].cuda() * percent * fisheye_K[0, 2]
        fisheye_K[1, 2] += torch.randn(1)[0].cuda() * percent * fisheye_K[1, 2]
        
        fisheye_D += torch.randn(4).cuda() * percent * fisheye_D

    ee_init_trans = ee_data['transl'][0]
    ee_init_rot = R.from_rotvec(ee_data['global_orient'][0]).as_matrix()
    ee_init = create_pose(ee_init_rot, ee_init_trans)
    gripper_init = is_closed[0]

    obj_init_trans = obj_data['transl'][0]
    obj_init_rot = R.from_rotvec(obj_data['global_orient'][0]).as_matrix()
    obj_init = create_pose(obj_init_rot, obj_init_trans)

    gripper_closed_val = np.median(drive_joint[is_closed == 1])

    fisheye_info = {
        'K': fisheye_K,
        'D': fisheye_D,
        'dims': fisheye_dims,
    }

    env_data = {
        'ee_init': ee_init,
        'gripper_init': gripper_init,
        'obj_init': obj_init,
        'scene_gaussians': scene_gaussians,
        'obj_gaussians': obj_gaussians,
        'ee_gaussians': ee_gaussians,
        'gripper_closed_val': gripper_closed_val,
        'robot': robot,
        'seg_ind_dict': seg_ind_dict,
        'M_ee2cam': M_ee2cam,
        'fisheye_info': fisheye_info,
        'min_drive_joint': min_drive_joint
    }

    env = gymnasium.make('splat_env/SplatWorld-v0', env_data=env_data)

    obs, _ = env.reset()
    obs_vis = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
    cv2.imshow('obv_vis', obs_vis)
    cv2.waitKey(0)

    prev_obs = obs
    episode_start_pose = None
    with torch.no_grad():
        policy.reset()
    for ind in range(100):
        rgb = np.stack([prev_obs['rgb'], obs['rgb']])

        camera_obs = dict()
        camera_obs[f'camera{0}_rgb'] = rgb
        
        obs_data = dict(camera_obs)

        eef_pos = np.stack([prev_obs['ee_pose'][0:3, 3],
                            obs['ee_pose'][0:3, 3]])
        eef_rot = np.stack([prev_obs['ee_pose'][0:3, 0:3],
                            obs['ee_pose'][0:3, 0:3]])
        eef_aa = R.from_matrix(eef_rot).as_rotvec()

        robot_obs = {
                f'robot{0}_eef_pos': eef_pos,
                f'robot{0}_eef_rot_axis_angle': eef_aa
            }
        obs_data.update(robot_obs)

        eef_is_closed = np.array([prev_obs['ee_gripper'], obs['ee_gripper']])[:, None]
        gripper_obs = {
                f'robot{0}_gripper_closed': eef_is_closed
            }
        obs_data.update(gripper_obs)

        # if episode_start_pose is None:
        #     episode_start_pose = list()
        #     for robot_id in range(1):
        #             pose = np.concatenate([
        #                 obs_data[f'robot{robot_id}_eef_pos'],
        #                 obs_data[f'robot{robot_id}_eef_rot_axis_angle']
        #             ], axis=-1)[-1]
        #             episode_start_pose.append(pose)

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
        
        # for action_ind in range(0, 9):
        for action_ind in range(1, 7):
            prev_obs = obs
        
            aa = action[action_ind]

            # env needs it to round rn
            gripper_action = np.clip(aa[-1], 0, 1).round()
            # gripper_action = np.clip(aa[-1], 0, 1)

            step = {
                'ee_pose': aa[0:6],
                'ee_gripper': gripper_action,
            }

            step_res = env.step(step)
            obs = step_res[0]

            obs_vis = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
            cv2.imshow('obv_vis', obs_vis)
            cv2.waitKey(10)
            
        #breakpoint()
    ###