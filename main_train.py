
import torch

print(torch.__version__)
import os
from config import Config

opt = Config('training.yml')
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from dataset_RGB import create_data_loader,DataLoaderTrain_npz,DataLoaderVal_npz
from U_model import unet
import losses
import glob
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from thop import profile

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def main():
    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'models', session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    #data prepare


    # ====check VGG layers====
    opt.VGGLayers = [int(layer) for layer in list(opt.VGGLayers)]
    opt.VGGLayers.sort()
    # print(opt.VGGLayers)

    if opt.VGGLayers[0] < 1 or opt.VGGLayers[-1] > 4:
        raise Exception("Only support VGG Loss on Layers 1 ~ 4")
    opt.VGGLayers = [layer - 1 for layer in list(opt.VGGLayers)]  ## shift index to 0 ~ 3

    if opt.w_VGG > 0:
        ### Load pretrained VGG
        from vgg_networks.vgg import Vgg16
        VGG = Vgg16(requires_grad=False)
        VGG = VGG.cuda()

    ######### Model ###########
    model_restoration = unet.Restoration(3, 6, 3,opt)
    model_restoration.cuda()




    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                            eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        # path_chk_rest = os.path.join(model_dir, "model_deblurring.pth")

        print('path_chk_rest', path_chk_rest)
        utils.load_checkpoint(model_restoration, path_chk_rest[0])
        start_epoch = utils.load_start_epoch(path_chk_rest[0]) + 1

        utils.load_optim(optimizer, path_chk_rest[0])

        for i in range(0, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    ######### Loss ###########
    criterion_char = losses.CharbonnierLoss()
    # criterion_edge = losses.EdgeLoss()
    # criterion = nn.MSELoss()

    ######### DataLoaders ###########
    train_dataset = DataLoaderTrain_npz(opt.father_train_path, opt)
    train_loader= create_data_loader(train_dataset, opt)
    val_dataset = DataLoaderVal_npz(opt.father_val_path, opt)
    val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False, num_workers=4, drop_last=False,
                            pin_memory=True)
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    best_ssim = 0
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0

        model_restoration.train()
        for iteration, data in enumerate(tqdm(train_loader), 1):

            for param in model_restoration.parameters():
                param.grad = None

            input_img = data[0].cuda()
            input_event = data[1].cuda()
            input_target = data[2].cuda()
            restored = model_restoration(input_img, input_event)

            loss_char = criterion_char(restored, input_target)
            # loss_edge = criterion_edge(restored, input_target)
            loss = loss_char
            loss.backward(retain_graph=False)

            torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), 20)

            optimizer.step()
            epoch_loss += loss.item()
            # torch.cuda.empty_cache()

        #### Evaluation ####
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            for ii, data_val in enumerate(tqdm(val_loader), 0):
                input_img = data_val[0].cuda()
                input_event = data_val[1].cuda()
                input_target = data_val[2].cuda()
                with torch.no_grad():
                    restored = model_restoration(input_img, input_event)

                for res, tar in zip(restored, input_target):

                    res = torch.clamp(res, 0, 1)
                    input1 = res.cpu().numpy().transpose([1, 2, 0])
                    input2 = tar.cpu().numpy().transpose([1, 2, 0])
                    ssim_rgb = SSIM(input1, input2, channel_axis=-1,data_range=1)
                    ssim_val_rgb.append(ssim_rgb)

                    psnr_rgb = PSNR(input1, input2)
                    psnr_val_rgb.append(psnr_rgb)

            ssim_val_rgb = np.mean(ssim_val_rgb)
            psnr_val_rgb = np.mean(psnr_val_rgb)

            with open(model_dir + '/BEST.txt', 'a') as f:
                f.write('Epoch:' + str(epoch) + ' PSNR:' + str(psnr_val_rgb) + ' ' + 'SSIM: ' + str(
                    ssim_val_rgb) + "\n")
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))

#
if __name__ == '__main__':
    main()


