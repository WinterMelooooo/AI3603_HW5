import argparse
from torchvision.transforms import Resize
import os
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_folder', type=str, default='', help='Src folder')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_parser()
    src_folder = opt.src_folder
    resizer = Resize((224, 224))
    for file in os.listdir(src_folder):
        img = Image.open(os.path.join(src_folder, file))
        img = resizer(img)
        img.save(os.path.join(src_folder, file))