_base_ = ['/mnt/petrelfs/sunzeyi/PixArt-alpha/configs/PixArt_xl2_internal.py']
data_root = '/home/sunzeyi/nvme/PixArt-alpha/workspace'

image_list_json = ['data_info.json',]

data = dict(type='InternalData', root='NOT_USED', image_list_json=image_list_json, transform='default_train', load_vae_feat=True)
image_size = 512

# model setting
window_block_indexes = []
window_size=0
use_rel_pos=False
model = 'PixArt_XL_2'
fp32_attention = True
load_from = "output/pretrained_models/PixArt-XL-2-512x512.pth"
vae_pretrained = "output/pretrained_models/sd-vae-ft-ema"
lewei_scale = 1.0

# training setting
use_fsdp = False   # if use FSDP mode
num_workers = 16
train_batch_size = 32 # 32
num_epochs = 30 # 3
gradient_accumulation_steps = 1
grad_checkpointing = False
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=1e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=10000)
eval_sampling_steps = 100
log_interval = 25
save_model_epochs = 3
save_model_steps = 100000
work_dir = 'output/debug'
