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
from pathlib import Path
import json
import shutil

from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.multi_uvc_camera import MultiUvcCamera
from umi.real_world.video_recorder import VideoRecorder
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from umi.common.precise_sleep import precise_wait

from oculus_reader.oculus_controller import VRPolicy 

import gymnasium as gym
import xarm_env

EXECUTION_LEAD_TIME = 0.150

class DemoRecorder:
    def __init__(self, base_dir="teleop_data"):
        self.base_dir = Path(base_dir)
        self.temp_dir = self.base_dir / "temp"
        self.success_dir = self.base_dir / "success"
        self.failed_dir = self.base_dir / "failed"

        for d in [self.temp_dir, self.success_dir, self.failed_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.recording = False
        self.waiting_for_label = False
        self.demo_path = None
        self.frames_path = None
        self.metadata = []
        self.frame_idx = 0

    def write_metadata(self, data_dict):
        self.metadata.append(data_dict)

    def start_recording(self, timestamp):
        self.demo_path = self.temp_dir / f"demo_{timestamp}"
        self.frames_path = self.demo_path / "frames"
        self.frames_path.mkdir(parents=True, exist_ok=True)

        self.recording = True
        self.waiting_for_label = False
        self.metadata = []
        self.frame_idx = 0
        print(f"[Recorder] Started new demonstration at {self.demo_path}")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.waiting_for_label = True

        metadata_file = self.demo_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print("[Recorder] Recording stopped. Waiting for success/failure...")

    def discard_unlabeled(self):
        if self.waiting_for_label:
            print("[Recorder] Movement restarted — discarding unlabeled demo.")
            # shutil.rmtree(self.demo_path)
            self.waiting_for_label = False

    def finalize(self, success=False, failure=False):
        if not self.waiting_for_label:
            return

        if success:
            dst = self.success_dir / self.demo_path.name
        elif failure:
            dst = self.failed_dir / self.demo_path.name
        else:
            print("[Recorder] No label given. Discarding demo.")
            shutil.rmtree(self.demo_path)
            self.waiting_for_label = False
            return

        shutil.move(str(self.demo_path), str(dst))
        print(f"[Recorder] Saved demo to: {dst}")
        self.waiting_for_label = False

# reset elego devices and get paths
reset_all_elgato_devices()
time.sleep(0.5)
v4l_paths = get_sorted_v4l_paths()

# these are for visualizations
# should create a 960 x 720 image from the gopro output
rw, rh, col, row = optimal_row_cols(
            n_cameras=len(v4l_paths),
            in_wh_ratio=4/3,
            max_resolution=(960, 960)
        )

# go pro capture settings
# got these from original umi
# even though gopro doesn't record at this resolution
# but I guess this is what the capture card outputs
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

# visualization transform
def vis_tf(data, input_res=res):
    img = data['color']
    f = get_image_transform(
        input_res=input_res,
        output_res=(rw,rh),
        bgr_to_rgb=False
    )
    img = f(img)

    data['color'] = img
    return data
vis_transform =[vis_tf]

def tf(data, input_res=res):
    img = data['color']
    f = get_image_transform(
        input_res=input_res,
        output_res=(rw,rh),
        bgr_to_rgb=True
    )
    img = f(img)

    data['color'] = img
    return data
transform =[tf]

# camera params
max_obs_buffer_size = 60
camera_obs_latency = 0.125
enable_multi_cam_vis = True

# just to match for now
# frequency = 10
# CHANGED
frequency = 50
dt = 1 / frequency

def run(robot_env, vr_policy):
    with SharedMemoryManager() as shm_manager:
        camera = MultiUvcCamera(
        dev_video_paths=v4l_paths,
        shm_manager=shm_manager,
        resolution=resolution,
        capture_fps=capture_fps,
        put_downsample=False,
        get_max_k=max_obs_buffer_size,
        receive_latency=camera_obs_latency,
        cap_buffer_size=cap_buffer_size,
        #transform=transform,
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

        _, _ = robot_env.reset()
        robot_env_unwrapped = robot_env.unwrapped

        demo_recorder  = DemoRecorder()

        # waits for camera data
        time.sleep(1.0)
        last_camera_data = None

        frame_latency = 1 / 60 # different than frame_dt I suppose
        iter_idx = 0
        t_start = time.monotonic()

        # start_time = time.time()
        # demo_recorder.start_recording(start_time)
        # video_paths = [str(demo_recorder.frames_path) + '/vid.mp4']
        # camera.restart_put(start_time=start_time)
        # camera.start_recording(video_path=video_paths, start_time=start_time)

        try:
            while True:
                if iter_idx % 100 == 0 and iter_idx != 0:
                    print(f'{(time.monotonic() - t_start) / (iter_idx )}')

                t_cycle_end = t_start + (iter_idx + 1) * dt

                # TODO these could have been gotten at different timestamps
                # may need to synchronize
                vr_info = vr_policy.get_info()

                # TODO there is a chance between get_info and next step 
                # movement was released so just no suden umps

                # camera data
                # k = 1
                # last_camera_data = camera.get(k=k, out=last_camera_data)

                # robot data
                robot_obs = robot_env_unwrapped._get_obs()

                # # img_to_use = last_camera_data[0]['color'][-1]
                # img_receive_timestamp = last_camera_data[0]['camera_receive_timestamp'][-1]

                ee_pose = robot_obs['ee_pose'][-1]
                ee_gripper = robot_obs['ee_gripper'][-1]
                robot_receive_timestamp = robot_obs["robot_receive_timestamp"][-1]
                gripper_receive_timestamp = robot_obs["gripper_receive_timestamp"][-1]

                movement_enabled = vr_info["movement_enabled"]
                success = vr_info["success"]
                failure = vr_info["failure"]

                # step robot
                if movement_enabled:
                    succ, action = vr_policy.forward(robot_obs, include_info=False)
                    if not succ:
                        continue

                    now = time.time()
                    target_time = now + EXECUTION_LEAD_TIME

                    action = {
                        'ee_pose': np.array([action['ee_pose']]),
                        'ee_gripper': np.array([action['ee_gripper']]),

                        'timestamps': np.array([target_time]),
                    }
                    _, _, _, _, _ = robot_env.step(action)


                if movement_enabled and demo_recorder.waiting_for_label:
                    demo_recorder.discard_unlabeled()
                
                # TODO should maybe have a target start time and 
                # start the video first then recording
                if movement_enabled and not demo_recorder.recording:
                    print('Starting Record')
                    start_time = time.time()

                    demo_recorder.start_recording(start_time)
                    video_paths = [str(demo_recorder.frames_path) + '/vid.mp4']
                    camera.restart_put(start_time)
                    camera.start_recording(video_path=video_paths, start_time=start_time)

                if demo_recorder.recording:
                    # demo_recorder.write_frame(img_to_use)
                    demo_recorder.write_metadata({
                        # image timing
                        "start_time": start_time,

                        # robot state
                        "ee_pose": ee_pose.tolist(),
                        "ee_gripper": float(ee_gripper),
                        "robot_receive_timestamp": robot_receive_timestamp,
                        "gripper_receive_timestamp": gripper_receive_timestamp,
                    })

                if not movement_enabled and demo_recorder.recording:
                    print('Stop Recording')
                    camera.stop_recording()
                    demo_recorder.stop_recording()

                if demo_recorder.waiting_for_label:
                    if success:
                        demo_recorder.finalize(success=True)
                    elif failure:
                        demo_recorder.finalize(failure=True)

                # here looks like subtract frame latency
                iter_idx += 1
                precise_wait(t_cycle_end - frame_latency)

        except KeyboardInterrupt:
            print("Interrupted")

            # camera.stop_recording()
            # demo_recorder.stop_recording()

        if multi_cam_vis is not None:
            multi_cam_vis.stop(wait=False)
        camera.stop(wait=False)

        camera.stop_wait()
        if multi_cam_vis is not None:
            multi_cam_vis.stop_wait()

if __name__ == "__main__":
    robot_env = gym.make(
        "xarm_servo_env/Xarm-v0",
        env_data={'ip': '192.168.1.211',
                  
                  "ee_controller_frequency": 200.0,
                  "ee_log_freq": None,
                  'ee_max_pos_speed': 2.0,
                  'ee_max_rot_speed': 6.0,
                  "ee_history_len": 200,

                  "gripper_controller_frequency": 60.0,
                  'gripper_log_freq': None,
                  "gripper_history_len": 60,

                  'compensate_latency': True,
                  'robot_action_latency': 0.1,
                  'gripper_action_latency': 0.1,

                  'thresh_closed': True,
                  }
    )

    vr_policy = VRPolicy(
        right_controller=True,
        rmat_reorder=[-2, -1, -3, 4],
        log_freq=None
    )

    run(robot_env, vr_policy)


    
