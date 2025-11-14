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
import xarm_env

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
    read_info,
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
    ckpt_path = '/home/hfreeman/harry_ws/repos/pruner_track/submodules/universal_manipulation_interface/data/outputs/2025.11.11/01.23.35_train_diffusion_unet_timm_umi/checkpoints/epoch=0110-train_loss=0.011.ckpt'
    data_dir = "/home/hfreeman/harry_ws/repos/pruner_track/datasets/DEMOS/chili_place_exp/demos/GX019722"

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
    gripper_splat_path = os.path.join(ASSET_DIR, 'gripper_splat.ply')
    ee_gaussians.load_ply(gripper_splat_path)

    gripper_segment_path = os.path.join(ASSET_DIR, 'segmented_gripper.pkl')
    segment_info = read_pickle(gripper_segment_path)
    seg_ind_dict = segment_info['label_dict']

    urdf_path = os.path.join(ASSET_DIR, 'gripper_gopro.urdf')
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
    M_ee2cam = create_pose(ee_2_cam_rot, ee_2_cam_trans)

    exp_info = read_info(data_dir)
    fisheye_calib_dir = os.path.join(GOPRO_CALIB_DIR, exp_info['fisheye_calib'])
    fisheye_camera_info = read_cam_info(fisheye_calib_dir)
    fisheye_K = torch.FloatTensor(np.array(fisheye_camera_info['K']).reshape(3, 3)).cuda()
    fisheye_D = torch.FloatTensor(np.array(fisheye_camera_info['D'])).cuda()
    fisheye_dims = [fisheye_camera_info['height'], fisheye_camera_info['width']]

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
    }

    env = gymnasium.make('splat_env/SplatWorld-v0', env_data=env_data)

    # robot env
    robot_env_data = {
        'api': '192.168.1.212'
    }
    robot_env = gymnasium.make('xarm_env/Xarm-v0', env_data=robot_env_data)
    robot_obs, _ = robot_env.reset()
    robot_init_trans = robot_obs['ee_pose'][0:3]
    robot_init_aa = robot_obs['ee_pose'][3:6]
    robot_init_rot = R.from_rotvec(robot_init_aa).as_matrix()
    robot_init_pose = create_pose(robot_init_rot, robot_init_trans)
    #

    obs, _ = env.reset()

    obs_init_pose = obs['ee_pose']
    M_splat2robot = robot_init_pose @ np.linalg.inv(obs_init_pose)

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
        
        for action_ind in range(1, 9):
            prev_obs = obs
        
            aa = action[action_ind]

            gripper_action = np.clip(aa[-1], 0, 1).round()

            ee_pose_aa = aa[0:6]
            ee_trans = ee_pose_aa[0:3]
            ee_aa = ee_pose_aa[3:]
            ee_rot = R.from_rotvec(ee_aa).as_matrix()
            ee_pose = create_pose(ee_rot, ee_trans)
            robot_pose = M_splat2robot @ ee_pose
            robot_trans = robot_pose[0:3, 3]
            robot_rot = robot_pose[0:3, 0:3]
            robot_aa = R.from_matrix(robot_rot).as_rotvec()
            robot_pose = np.array(robot_trans.tolist() + robot_aa.tolist())
            robot_action = {
                'ee_pose': robot_pose,
                'ee_gripper': gripper_action,
            }
            robot_obs = robot_env.step(action=robot_action)[0]

            robot_trans_update = robot_obs['ee_pose'][0:3]
            robot_rot_update = R.from_rotvec(robot_obs['ee_pose'][3:]).as_matrix()
            robot_pose_update = create_pose(robot_rot_update, robot_trans_update)
            ee_pose_update = np.linalg.inv(M_splat2robot) @ robot_pose_update
            ee_trans_update = ee_pose_update[0:3, 3]
            ee_aa_update = R.from_matrix(ee_pose_update[0:3, 0:3]).as_rotvec()
            ee_pose_update = np.array(ee_trans_update.tolist() + ee_aa_update.tolist())
            step = {
                'ee_pose': ee_pose_update,
                'ee_gripper': gripper_action,
            }

            # step = {
            #     'ee_pose': aa[0:6],
            #     'ee_gripper': gripper_action,
            # }

            step_res = env.step(step)
            obs = step_res[0]

            obs_vis = cv2.cvtColor(obs['rgb'], cv2.COLOR_RGB2BGR)
            cv2.imshow('obv_vis', obs_vis)
            cv2.waitKey(1)

            # obs_pose = obs['ee_pose']
            # robot_pose = M_splat2robot @ obs_pose
            # robot_trans = robot_pose[0:3, 3]
            # robot_rot = robot_pose[0:3, 0:3]
            # robot_aa = R.from_matrix(robot_rot).as_rotvec()
            # robot_pose = np.array(robot_trans.tolist() + robot_aa.tolist())
            # robot_action = {
            #     'ee_pose': robot_pose,
            #     'ee_gripper': obs['ee_gripper'],
            # }
            # robot_env.step(action=robot_action)
            
        #breakpoint()
    ###