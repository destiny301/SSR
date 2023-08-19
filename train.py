import os
import argparse
from sklearn.metrics import confusion_matrix, precision_recall_curve, mean_absolute_error
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch import optim

from model.ssr import SSR, SSR_wo_conv
from data.srloader import SRData
import cv2

from utils.img_util import tensor2img
from utils.psnr_ssim import calculate_psnr, calculate_ssim

def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer

def main(args):
    print(args)

    # load data
    print('============================loading data============================')
    root = os.path.join('../datasets', args.data) # dataset path
    dataset_tr = SRData(root, args, 'train')
    dataset_te = SRData(root, args, 'val')
    train_loader = DataLoader(dataset_tr, args.batchsz, num_workers=4, shuffle=True)
    test_loader = DataLoader(dataset_te, 1, num_workers=4, shuffle=False)

    # check cuda
    device = torch.device(args.dev if torch.cuda.is_available() else 'cpu')
    print('training device:', device)

    # build model
    num_cls = 1 if args.loss == 'bce' else 2
    if args.conv:
        model = SSR(args, num_cls)
    else:
        model = SSR_wo_conv(args, num_cls)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion = nn.BCEWithLogitsLoss() if args.loss == 'bce' else nn.CrossEntropyLoss()
    l1loss = nn.L1Loss()

    # results saving path
    resultfolder = os.path.join('./results', args.data)
    if not os.path.exists(resultfolder):
        os.mkdir(resultfolder)

    modelfolder = os.path.join(resultfolder, 'checkpoint')
    if not os.path.exists(modelfolder):
        os.mkdir(modelfolder)
    pretrain = '_pretrained' if args.pretrain else '_fs'
    modelname = 'ssr' if args.conv else 'ssr_wo_conv'
    modelpath = os.path.join(modelfolder, modelname+pretrain+'_final.pth')
    
    # train and validate
    print('============================Training============================')
    for epoch in range(args.epoch):
        train_loss, test_loss = 0.0, 0.0
        cm = np.zeros((2, 2))

        # train model
        for xtr, ytr, sr in train_loader:
            a3tr = nn.MaxPool2d(kernel_size=(args.imgsz//args.token_size), stride=(args.imgsz//args.token_size))(ytr)
            a2tr = nn.MaxPool2d(kernel_size=(args.imgsz//(args.token_size*2)), stride=(args.imgsz//(args.token_size*2)))(ytr)
            a1tr = nn.MaxPool2d(kernel_size=(args.imgsz//(args.token_size*4)), stride=(args.imgsz//(args.token_size*4)))(ytr)

            xtr, ytr, sr = xtr.to(device), ytr.to(device), sr.to(device)
            a3tr, a2tr, a1tr = a3tr.to(device), a2tr.to(device), a1tr.to(device)

            optimizer.zero_grad()
            psr, p3tr, p2tr, p1tr = model(xtr) # sr_img, class for 4*4, 8*8, 16*16 tiles

            ltr = criterion(p3tr, torch.squeeze(a3tr).long() if args.loss == 'ce' else a3tr)
            if args.pyramid:
                ltr += criterion(p2tr, a2tr)
                ltr += criterion(p1tr, a1tr)
            ltr += args.alpha*l1loss(psr, sr)
            ltr.backward()
            optimizer.step()
            train_loss += ltr.item()
        
        # evaluate model
        if args.eval:
            img_dir = os.path.join(resultfolder, 'images')
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            lr_path = os.path.join(img_dir, 'lr')
            if not os.path.exists(lr_path):
                os.mkdir(lr_path)
            hr_path = os.path.join(img_dir, 'hr')
            if not os.path.exists(hr_path):
                os.mkdir(hr_path)
            sr_path = os.path.join(img_dir, 'epoch'+str(epoch))
            if not os.path.exists(sr_path):
                os.mkdir(sr_path)

            pred = np.zeros((0,1,args.token_size,args.token_size))
            gt = np.zeros((0,1,args.token_size,args.token_size))
            psnr, ssim = 0.0, 0.0
            with torch.no_grad():
                for xte, yte, sr, name in test_loader:
                    a3te = nn.MaxPool2d(kernel_size=(args.imgsz//args.token_size), stride=(args.imgsz//args.token_size))(yte)
                    xte, yte, a3te = xte.to(device), yte.to(device), a3te.to(device)

                    # feed data and get output
                    psr, p3te, _, _ = model(xte)
                    
                    # convert tensor to image for saving and evaluation
                    psr_img = tensor2img(psr)
                    hr_img = tensor2img(sr)
                    xte = nn.Upsample(scale_factor=4, mode='bicubic')(xte)
                    lr_img = tensor2img(xte)

                    if epoch == 0:
                        cv2.imwrite(os.path.join(lr_path, name[0]+'.png'), lr_img)
                        cv2.imwrite(os.path.join(hr_path, name[0]+'.png'), hr_img)
                    cv2.imwrite(os.path.join(sr_path, name[0]+'.png'), psr_img)

                    # evaluate TS (maxF and TPR) and TR (SSIM and PSNR), separately
                    psnr += calculate_psnr(img=psr_img, img2=hr_img, crop_border=4, test_y_channel=True)
                    ssim += calculate_ssim(img=psr_img, img2=hr_img, crop_border=4, test_y_channel=True)
                    
                    lte = criterion(p3te, torch.squeeze(a3te).long() if args.loss == 'ce' else a3te)
                    test_loss += lte.item()
                    
                    if args.loss != 'bce':
                        p3te = torch.unsqueeze(torch.argmax(p3te, 1), 1)

                    p3te = p3te.cpu().numpy()
                    a3te = a3te.cpu().numpy()
                    a3te[a3te>=0.5] = 1
                    a3te[a3te<0.5] = 0
                    pred = np.append(pred, p3te, axis=0)
                    gt = np.append(gt, a3te, axis=0)
                    p3te[p3te>=args.th] = 1
                    p3te[p3te<args.th] = 0
                    cm += confusion_matrix(a3te.astype(np.int32).flatten(), p3te.flatten())
                xte = xte.cpu().numpy()
                sr = sr.cpu().numpy()
                psr = psr.clamp_(0, 1)
                psr = psr.cpu().numpy()
            pred = pred.flatten()
            gt = gt.flatten()
            precision, recall, threshold = precision_recall_curve(gt, pred)
            f_scores = 1.3*recall*precision/(recall+0.3*precision+ 1e-20)

            mae = mean_absolute_error(gt, pred)
            tn, fp, fn, tp = cm.ravel()

            print('epoch', epoch+1, '\ttrain loss:', "{:.4f}".format(train_loss/len(train_loader)), '\ttest loss', "{:.4f}".format(test_loss/len(test_loader)),'\tmaxf:', "{:.4f}".format(np.max(f_scores)), '\tIoU:', "{:.4f}".format(tp/(tp+fn+fp+ 1e-20)), '\tTPR:', "{:.4f}".format(tp/(tp+fn+ 1e-20)),
                '\tPSNR:', "{:.4f}".format(psnr/len(test_loader)), '\tSSIM:', "{:.4f}".format(ssim/len(test_loader)))

        torch.save(model.state_dict(), modelpath)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, help='which dataset to use(coco2017/MSRA10K/bdd100k)', default='bdd100k')
    argparser.add_argument('--imgsz', type=int, help='image size(low-resolution)', default=256)
    argparser.add_argument('--aug', action='store_true', help='data augmentation or not')
    argparser.add_argument('--norm', action='store_true', help='normalize or not')
    argparser.add_argument('--token_size', type=int, help='number of tiles (eg. 4*4 or 8*8)', default=4)
    argparser.add_argument('--scale', type=int, help='up scale ratio', default=4)

    argparser.add_argument('--pretrain', action='store_true', help='load pretrained TR module or not')
    argparser.add_argument('--ckpt', type=str, help='checkpoint path for pretrained TR module', default='.')
    argparser.add_argument('--eval', action='store_true', help='evaluate during training or not')
    argparser.add_argument('--conv', action='store_true', help='conv layers for negative tiles or not')
    argparser.add_argument('--dim', type=int, help='attention embedding dimension for tile selection', default=96)
    argparser.add_argument('--patchsz', type=int, help='patchsz for TS Module embedding', default=2)
    argparser.add_argument('--th', type=float, help='threshold for attention or not', default=0.01)
    argparser.add_argument('--dev', type=str, help='cuda device', default='cuda:0')
    argparser.add_argument('--epoch', type=int, help='number of training epochs', default=50)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.00001)
    argparser.add_argument('--lr_slot', type=int, help='learning rate change point(related to batch size)', default=2000)
    argparser.add_argument('--batchsz', type=int, help='batch size', default=2)

    argparser.add_argument('--pyramid', action='store_true', help='use pyramid structure or not')
    argparser.add_argument('--loss', type=str, help='loss function(bce/ce)', default='bce')
    argparser.add_argument('--alpha', type=float, help='TS/TR loss ratio', default=1)

    args = argparser.parse_args()
    main(args)
