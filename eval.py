# evaluate generated images with PSNR/SSIM/FID/KID/KYD

import os, cv2, shutil
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from cleanfid import fid
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any

from utils.img_util import tensor2img
from utils.psnr_ssim import calculate_psnr, calculate_ssim
from utils.kyd import compute_KYD


class Datasets(Dataset):
    def __init__(self, dir1, dir2) -> None:
        super().__init__()
        
        self.hr_folder = dir1
        self.sr_folder = dir2

        self.hr_files = os.listdir(self.hr_folder)
        self.sr_files = os.listdir(self.sr_folder)

    def __len__(self):
        return len(self.sr_files)
    
    def __getitem__(self, index: Any):
        hr_imgpath = os.path.join(self.hr_folder, self.sr_files[index])
        x = cv2.imread(hr_imgpath)
        hr = cv2.resize(x, [1024, 1024], interpolation=cv2.INTER_CUBIC)
        hr = hr/(hr.max()+10e-8)
        hr = np.transpose(hr, (2, 0, 1))

        sr_imgpath = os.path.join(self.sr_folder, self.sr_files[index])
        x = cv2.imread(sr_imgpath)
        x = x/(x.max()+10e-8)
        sr = np.transpose(x, (2, 0, 1))
        return hr, sr

# compute PSNR and SSIM
def get_psnr_ssim(dir1, dir2):
    dataset = Datasets(dir1, dir2)
    dtloader = DataLoader(dataset, 1, num_workers=4, shuffle=True)

    ps, ss = 0.0, 0.0
    for hr, sr in tqdm(dtloader):
        hr_img = tensor2img(hr)
        sr_img = tensor2img(sr)
        ps += calculate_psnr(img=sr_img, img2=hr_img, crop_border=4, test_y_channel=True)
        ss += calculate_ssim(img=sr_img, img2=hr_img, crop_border=4, test_y_channel=True)
    return ps/len(dtloader), ss/len(dtloader)
    

# compute FID/KID/KYD
def compute_distance(hr_folder, sr_folder):

    fid_sr = fid.compute_fid(hr_folder, sr_folder)
    kid_sr = fid.compute_kid(hr_folder, sr_folder)
    
    kyd_sr = compute_KYD(hr_folder, sr_folder)
    
    return fid_sr, kid_sr, kyd_sr

if __name__=='__main__':
    hr_folder = os.path.join('./results/bdd100k/images/', 'hr')
    sr_folder = os.path.join('./results/bdd100k/images/', 'sr')

    psnr, ssim = get_psnr_ssim(hr_folder, sr_folder)
    fid_sr, kid_sr, kyd_sr = compute_distance(hr_folder, sr_folder)

    print('psnr_sr:', psnr)
    print('ssim_sr:', ssim)
    print('FID score(sr):', fid_sr)
    print('KID score(sr):', kid_sr)
    print('KYD score(sr):', kyd_sr)


    