from model import Unet,GaussianDiffusion
from trainer import Trainer
import torch
from datetime import datetime
import os
import argparse
torch.backends.cudnn.benchmark = True
# torch.manual_seed(4096)

# if torch.cuda.is_available():
#   torch.cuda.manual_seed(4096)
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str, default='No comment', help='Comment to add to the run')
    parser.add_argument('--data_dir', type=str, default='../data/faces', help='data directory')
    parser.add_argument('--results_folder', type=str, default='./resultsCAT', help='results folder')
    parser.add_argument('--img_size', type=int, default=64, help='image size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--train_num_steps', type=int, default=30000, help='number of training steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--grad_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='exponential moving average decay')
    parser.add_argument('--channels', type=int, default=16, help='number of channels of the first layer of CNN')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=(1, 2, 4), help='model size')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of steps')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='beta schedule')
    opt = parser.parse_args()
    return opt

opt = get_parser()
Comment = opt.comment
path = opt.data_dir
results_folder = opt.results_folder
timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
results_folder = os.path.join(results_folder, Comment, timestamp)
os.makedirs(results_folder, exist_ok=True)
IMG_SIZE = opt.img_size        # Size of images, do not change this if you do not know why you need to change
batch_size = opt.batch_size        # batch size
train_num_steps = opt.train_num_steps        # total training steps
lr = opt.lr                # learning rate
grad_steps = opt.grad_steps         # gradient accumulation steps, the equivalent batch size for updating equals to batch_size * grad_steps = 16 * 1
ema_decay = opt.ema_decay           # exponential moving average decay

channels = opt.channels             # Numbers of channels of the first layer of CNN
dim_mults = opt.dim_mults        # The model size will be (channels, 2 * channels, 4 * channels, 4 * channels, 2 * channels, channels)
timesteps = opt.timesteps         # Number of steps (adding noise)
beta_schedule = opt.beta_schedule       # Beta schedule

super_params = {
    'Comment': Comment,
    'IMG_SIZE': IMG_SIZE,
    'batch_size': batch_size,
    'train_num_steps': train_num_steps,
    'lr': lr,
    'grad_steps': grad_steps,
    'ema_decay': ema_decay,
    'channels': channels,
    'dim_mults': dim_mults,
    'timesteps': timesteps,
    'beta_schedule': beta_schedule
}
with open(f'{results_folder}/params.txt', 'w') as f:
    print(super_params, file=f)


model = Unet(
    dim = channels,
    dim_mults = dim_mults
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMG_SIZE,
    timesteps = timesteps,
    beta_schedule = beta_schedule
)

trainer = Trainer(
    diffusion,
    path,
    train_batch_size = batch_size,
    train_lr = lr,
    train_num_steps = train_num_steps,
    gradient_accumulate_every = grad_steps,
    ema_decay = ema_decay,
    save_and_sample_every = 1000,
    results_folder = results_folder,
    tsboard_frq = 20,
    timestamp = timestamp
)

print(trainer.device)
trainer.load("/home/yktang/AI3603_HW5/code/resultsCAT/LargeChannel_MedianDimMults_LargeBatchSize_LongEpoch/2024_12_22_14_19_32/model-12.pt")
# Train
trainer.train()

ckpt = f"{trainer.chkpt_folder}/model-30.pt"
trainer.load(ckpt)
# Random generation
trainer.inference(output_path=f"{results_folder}/submission")
# Fusion generation
trainer.inference2(lamda=0.5,index1=9000,index2=8888,output_path=f"{results_folder}/fusion",source_path=f'{results_folder}/source', data_dir=path)