import torch.nn as nn
import torch
import torch.nn.functional as F

from model.tile_refinement import TileRefinement, Upsample
from model.tile_selection import TileSelection
from model.net_blocks import window_partition, window_reverse

class SSR(nn.Module):
    '''
    feed pos tiles to TR Module, neg tiles to conv layers, then reconstruct them together
    '''
    def __init__(self, args, num_cls) -> None:
        super().__init__()
        self.select_model = TileSelection(args, num_cls)
        self.sr_model = TileRefinement(upscale=4, img_size=64,
                   window_size=16, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                   embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
        
        if args.pretrain:
            self.sr_model.load_state_dict(torch.load(args.ckpt)['params_ema'])
            print('----------loaded TR pretrained model-----------------')

        # image reconstruction
        self.conv_first = nn.Conv2d(3, 180, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(180, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(4, 64)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        patch_fea3, patch_fea2, patch_fea1 = self.select_model(x)
        pi_prime = F.gumbel_softmax(patch_fea3, hard=True) # B, 1, 4, 4
        patch_x = window_partition(x.permute(0, 2, 3, 1), window_size=(H//4)) # B*4*4, H/4, W/4, 3
        patch_x = patch_x.permute(0, 3, 1, 2)  # B*4*4, 3, H/4, W/4
        pi_prime = pi_prime.view(-1)
        
        # feature extraction
        lr_fea = torch.zeros((0, 180, 64, 64)).to(x.device)
        for i in range(B*16):
            if pi_prime[i] == 1:
                posX, fea = self.sr_model(patch_x[i].unsqueeze(0))
                lr_fea = torch.cat([lr_fea, fea], dim=0)
            else:
                fea = self.conv_first(patch_x[i].unsqueeze(0))
                lr_fea = torch.cat([lr_fea, fea], dim=0)
        lr_fea = window_reverse(lr_fea.permute(0, 2, 3, 1), window_size=H//4, H=H, W=W).permute(0, 3, 1, 2)

        # image reconstruction
        sr_fea = self.upsample(self.conv_before_upsample(lr_fea))
        sr = self.conv_last(sr_fea)

        return sr, patch_fea3, patch_fea2, patch_fea1
    

class SSR_wo_conv(nn.Module):
    '''
    simply feed pos tiles to TR module, neg tiles to upsample layer
    '''
    def __init__(self, args, num_cls) -> None:
        super().__init__()
        self.select_model = TileSelection(args, num_cls)
        self.sr_model = TileRefinement(upscale=4, img_size=64,
                   window_size=16, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                   embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle')
        
        if args.pretrain:
            self.sr_model.load_state_dict(torch.load(args.ckpt)['params_ema'])
            print('----------loaded TR pretrained model-----------------')

        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

    def forward(self, x):
        B, C, H, W = x.shape
        patch_fea3, patch_fea2, patch_fea1 = self.select_model(x)
        pi_prime = F.gumbel_softmax(patch_fea3, hard=True) # B, 1, 4, 4
        patch_x = window_partition(x.permute(0, 2, 3, 1), window_size=(H//4)) # B*4*4, H/4, W/4, 3
        patch_x = patch_x.permute(0, 3, 1, 2)  # B*4*4, 3, H/4, W/4
        pi_prime = pi_prime.view(-1)

        sr = torch.zeros((0, C, H, W)).to(x.device)

        for i in range(B*16):
            if pi_prime[i] == 1:
                posX, _ = self.sr_model(patch_x[i].unsqueeze(0))
                sr = torch.cat([sr, posX], dim=0)
            else:
                negX = self.upsample(patch_x[i].unsqueeze(0))
                sr = torch.cat([sr, negX], dim=0)
        sr = window_reverse(sr.permute(0, 2, 3, 1), window_size=H, H=H*4, W=W*4)

        return sr.permute(0, 3, 1, 2), patch_fea3, patch_fea2, patch_fea1
    