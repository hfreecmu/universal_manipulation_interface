import hydra
import torch
from torch.utils.data import DataLoader
import dill
import numpy as np
import matplotlib.pyplot as plt
import cv2

from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

ckpt_path = '/home/hfreeman/harry_ws/repos/pruner_track/submodules/universal_manipulation_interface/data/outputs/2025.11.10/15.19.54_train_diffusion_unet_timm_umi/checkpoints/epoch=0040-train_loss=0.019.ckpt'

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

cfg.val_dataloader['batch_size'] = 2

dataset: BaseImageDataset
dataset = hydra.utils.instantiate(cfg.task.dataset)
assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
val_dataset = dataset.get_validation_dataset()
val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

# val for no shuffle
train_dataloader = DataLoader(dataset, **cfg.val_dataloader)

# val_sampling_batch = next(iter(train_dataloader))
val_sampling_batch = next(iter(val_dataloader))

batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))

gt_action = batch['action']
pred_action = policy.predict_action(batch['obs'], None)['action_pred']

# B, T, _ = pred_action.shape
# pred_action = pred_action.view(B, T, -1, 10)
# gt_action = gt_action.view(B, T, -1, 10)

obs = batch['obs']
rgb = obs['camera0_rgb']

rgb_debug = rgb[0]
rgb_debug = (rgb_debug * 255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
debug_im = np.hstack((rgb_debug[0], rgb_debug[1]))
plt.imshow(debug_im)
plt.show()


gt_action_debug = gt_action[0]
pred_action_debug = pred_action[0]

breakpoint()
