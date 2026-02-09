import os
import time
import math
import cv2
import numpy as np

from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.multi_uvc_camera import MultiUvcCamera
from umi.real_world.video_recorder import VideoRecorder
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from umi.common.cv_util import draw_predefined_mask

from vine_prune.utils.paths import ASSET_DIR

reset_all_elgato_devices()
time.sleep(0.5)
v4l_paths = get_sorted_v4l_paths()

# this is for vis
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

    # img[gripper_seg_mask > 0] = 255
    # img[valid_mask == 0] = 0

    data['color'] = img
    return data
vis_transform =[vis_tf]

VALID_MASK = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'valid_area.png'), -1)
GRIPPER_SEG_MASK = cv2.imread(os.path.join(ASSET_DIR, 'gripper_masks', 'gripper_seg_mask.png'), -1)
mask_input_res = (960, 720)

# VALID_MASK = cv2.resize(VALID_MASK, (res[0], res[1]), interpolation=cv2.INTER_NEAREST)
# GRIPPER_SEG_MASK = cv2.resize(GRIPPER_SEG_MASK, (res[0], res[1]), interpolation=cv2.INTER_NEAREST)

def tf(data, input_res=res, mask_input_res=mask_input_res):
    img = data['color']

    # img = np.copy(img)
    # img[GRIPPER_SEG_MASK > 0] = 0
    # img[VALID_MASK == 0] = 0

    f = get_image_transform(
        input_res=input_res,
        output_res=(224, 224), 
        # obs output rgb
        bgr_to_rgb=True)
    img = np.ascontiguousarray(f(img))

    f_mask = get_image_transform(
        input_res=mask_input_res,
        output_res=(224, 224), 
        is_mask=True)
    valid_mask = np.ascontiguousarray(f_mask(VALID_MASK))
    gripper_seg_mask = np.ascontiguousarray(f_mask(GRIPPER_SEG_MASK))

    img[gripper_seg_mask > 0] = 255
    img[valid_mask == 0] = 0

    # img = draw_predefined_mask(img, color=(0,0,0), 
    #                         mirror=False, gripper=False, finger=True, use_aa=True)
    data['color'] = img
    return data
transform = [tf]

max_obs_buffer_size = 60
# camera_obs_latency = 0.125 #0.17
# camera_obs_latency = 0
# camera_obs_latency = 0.17
camera_obs_latency = 0.17

enable_multi_cam_vis = True

camera_obs_horizon=2
camera_down_sample_steps=1
frequency = 10

if True:
    with SharedMemoryManager() as shm_manager:
        # with MultiUvcCamera(
        #         dev_video_paths=v4l_paths,
        #         shm_manager=shm_manager,
        #         resolution=resolution,
        #         capture_fps=capture_fps,
        #         put_downsample=False,
        #         get_max_k=max_obs_buffer_size,
        #         receive_latency=camera_obs_latency,
        #         cap_buffer_size=cap_buffer_size,
        #         #transform=[tf],
        #         vis_transform=vis_transform,
        #         video_recorder=video_recorder,
        #         verbose=False
        #     ) as camera:
        
        #     multi_cam_vis = MultiCameraVisualizer(
        #         camera=camera,
        #         row=row,
        #         col=col,
        #         rgb_to_bgr=False
        #     )

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

        while True:
            # now let's get data
            k = math.ceil(
                camera_obs_horizon * camera_down_sample_steps \
                * (60 / frequency)) + 2 # they say 2 here is optional

            last_camera_data = camera.get(k=k, out=last_camera_data)

            cv2.imshow('test', np.copy(last_camera_data[0]['color'][-1]))
            cv2.waitKey(1)

        if multi_cam_vis is not None:
            multi_cam_vis.stop(wait=False)
        camera.stop(wait=False)

        camera.stop_wait()
        if multi_cam_vis is not None:
            multi_cam_vis.stop_wait()
        breakpoint()