import os
import time
import math
import cv2

from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.multi_uvc_camera import MultiUvcCamera
from umi.real_world.video_recorder import VideoRecorder
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer

def format_int(ind):
    return "{:0>6d}".format(ind)

# WARNING - assumes robot as started in ufactory studio

OUTPUT_DIR = '/home/hfreeman/Downloads/debug'

image_dir = os.path.join(OUTPUT_DIR, 'images')
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

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
    data['color'] = img
    return data
vis_transform =[vis_tf]

max_obs_buffer_size = 60
camera_obs_latency = 0.125 #0.17
# camera_obs_latency = 0

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
            transform=vis_transform,
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

        # waits for camera data and arm
        time.sleep(1.0)

        last_camera_data = None

        ind = 0
        while True:
            k = math.ceil(
                camera_obs_horizon * camera_down_sample_steps \
                * (60 / frequency)) + 2 # they say 2 here is optional

            last_camera_data = camera.get(k=k, out=last_camera_data)
            camera_data = last_camera_data[0]
            ee_im = camera_data['color'][-1]

            im_path = os.path.join(image_dir, f'frame_{format_int(ind)}.jpg')
            cv2.imwrite(im_path, ee_im)
            ind += 1

            time.sleep(0.5)

        if multi_cam_vis is not None:
            multi_cam_vis.stop(wait=False)
        camera.stop(wait=False)

        camera.stop_wait()
        if multi_cam_vis is not None:
            multi_cam_vis.stop_wait()
