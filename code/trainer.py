import math
from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

import torchvision
from torchvision import transforms as T, utils
import shutil
from tqdm.auto import tqdm
from ema_pytorch import EMA
from PIL import Image
from accelerate import Accelerator
import os
import glob
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter


##utils
def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


##Create dataset


class Dataset(Dataset):
    def __init__(self, folder, image_size):
        self.folder = folder
        self.image_size = image_size
        print(f'Loading images from {folder} with image size {image_size}')
        file_pattern = os.path.join(folder, 'cat_*.jpg')
        self.paths = glob.glob(file_pattern)
        #################################
        ## Optional: Data Augmentation ##
        #################################
        self.transform = T.Compose([T.Resize(image_size), T.ToTensor()])
        self.resize = Resize([224,224])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = self.resize(Image.open(path))
        return self.transform(img)


##Define Trainer: define the updating process


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./resultsCAT',
        split_batches=True,
        inception_block_idx=2048,
        tsboard_frq=20,
        timestamp = None
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision='no')
        self.tsboard_frq = tsboard_frq
        self.timestamp = timestamp
        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size)
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.chkpt_folder = os.path.join(results_folder, "checkpoints")
        os.makedirs(self.chkpt_folder, exist_ok=True)
        self.output_img_folder = os.path.join(results_folder, "output_images")
        os.makedirs(self.output_img_folder, exist_ok=True)
        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.writer = SummaryWriter(log_dir=os.path.join(results_folder,"runs"))
    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, os.path.join(self.chkpt_folder, f'model-{milestone}.pt'))

    def load(self, ckpt):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(ckpt, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    # region
                    """comments:"""
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    # endregion

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss_{self.timestamp}: {total_loss:.4f}')
                #if self.step % self.tsboard_frq == 0:
                self.writer.add_scalar('Loss/train', total_loss, self.step)
                # region
                """comments:"""
                accelerator.wait_for_everyone()
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()
                # endregion

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        # region
                        """comments:"""
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                        # endregion

                        all_images = torch.cat(all_images_list, dim=0)

                        utils.save_image(all_images, os.path.join(self.output_img_folder, f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples)))

                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def inference(self, num=1000, n_iter=100, output_path='./submission'):
        """comments:"""
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with torch.no_grad():
            for i in range(n_iter):
                batches = num_to_groups(num // n_iter, 10)
                all_images = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))[0]
                for j in range(all_images.size(0)):
                    torchvision.utils.save_image(all_images[j], f'{output_path}/{i * 10 + j + 1}.jpg')

    def inference2(self, num=1, n_iter=1, lamda=0.5, index1=1, index2=2, output_path='./fusion', source_path='./source', data_dir = None):
        """comments:"""
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with torch.no_grad():
            batches = num_to_groups(num // n_iter, 1)
            all_images = list(map(lambda n: self.ema.ema_model.sample2(batch_size=n, lamda=lamda, index1=index1, index2=index2, data_dir=data_dir), batches))[0]
            torchvision.utils.save_image(all_images, f'{output_path}/{index1}_{index2}.jpg')
            os.makedirs(source_path, exist_ok=True)
            shutil.copy(f'{data_dir}/cat_{index1:05d}.jpg', source_path)
            shutil.copy(f'{data_dir}/cat_{index2:05d}.jpg', source_path)
