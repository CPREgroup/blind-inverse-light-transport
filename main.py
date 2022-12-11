import os
import time
import sys
from functools import partial
import visdom
import imageio
from video_svd import *
from torch import optim
from torch.utils.data import DataLoader
import scipy
import json
import socket
import datetime
import random
from data_loading import *
from model_T import *
from rollingnet import Fullcnn
from train_opt import args_myParser
import torch
from tools import psnr, device
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

args = args_myParser()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuind

if __name__ == '__main__':
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    learning_rate = args.learning_rate
    data_dir = args.data_dir
    folder_name= args.folder
    dataset = args.dataset
    seq_name = args.seq_name
    skip_frames = args.skip_frames
    nvec = args.nvec
    machine_name = socket.gethostname()
    cur_time = datetime.datetime.now()
    created_outputdir = False

    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    frames_dict = json.load(open(os.path.join(args.data_dir, args.folder, "frames.txt"), 'r'))
    Z, overexp_mask, gt_imgs = load_observations(data_dir, folder_name, dataset, seq_name, skip_frames,
                                                 n_downsample=args.num_downsample, frames_dict=frames_dict)
    s_i = 16  # latent video height
    s_j = 16  # ... width
    s_I = Z.shape[1]  # observed video height
    s_J = Z.shape[2]  # ... width
    s_c = Z.shape[3]  # color channels in either video
    s_b = 1  # batch size (no foreseeable reason why this would be other than 1 but let's keep it around)
    s_t = Z.shape[0]

    sv_U, sv_S, sv_V = video_svdbasis(Z[:, :, :, 0], k=nvec)
    sv_U = np.expand_dims(sv_U, 0)
    sv_S = np.expand_dims(sv_S, 0)
    sv_V = np.expand_dims(sv_V, 0)
    # if there are more channels, concatenate
    for c in range(Z.shape[3] - 1):
        sv_Uc, sv_Sc, sv_Vc = video_svdbasis(Z[:, :, :, c + 1], k=nvec)
        sv_U = np.concatenate((sv_U, np.expand_dims(sv_Uc, 0)), 0)
        sv_S = np.concatenate((sv_S, np.expand_dims(sv_Sc, 0)), 0)
        sv_V = np.concatenate((sv_V, np.expand_dims(sv_Vc, 0)), 0)

    # # A bit of additional processing to multiply in the half-power of the singular values and to shape into TNet's assumed format
    # # In a notational deviation from the paper, we happen to be working with transposes of the matrices so it's the V vectors we want.
    sv_V_aux = sv_V[:, :nvec, :]
    sv_V_aux = np.reshape(sv_V_aux, [s_c, nvec, s_I, s_J])  # [c, sv, I, J]
    sv_V_aux = sv_V_aux * np.reshape(np.sqrt(sv_S / np.expand_dims(sv_S[:, 1], 1)), [s_c, nvec, 1, 1])
    svecs = torch.from_numpy(sv_V_aux).float().to(device)

    Z = torch.from_numpy(Z).float().to(device)
    Zmean = torch.mean(Z, dim=0)
    ZT = Z.permute(3,1,2,0).view(s_c,s_I,s_J,s_t).unsqueeze(0)

    oemask = 1.0 - overexp_mask.astype(np.float32)
    # Also for excluding finite differences in smoothness priors:
    oemask_dI = oemask[1:, :] * oemask[:-1, :]
    oemask_dJ = oemask[:, 1:] * oemask[:, :-1]

    oemask = torch.from_numpy(oemask).float().to(device)
    oemask_dI = torch.from_numpy(oemask_dI).float().to(device)
    oemask_dJ = torch.from_numpy(oemask_dJ).float().to(device)

    pos = torch.from_numpy(positionencoding1D(1,1)).to(device)
    pos = pos.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,s_i,s_j,s_t)

    criterion1 = nn.L1Loss()
    criterion = nn.MSELoss()

    logname = str(cur_time.strftime("%Y-%m-%d-%H%M%S"))

    tnet = TNet_SV(svecs=svecs, Zmean=Zmean, out_channels=s_c).to(device)
    VNet = Fullcnn(Z,pos,s_t,args).to(device)

    optimizer = torch.optim.Adam([{'params': VNet.parameters(), 'lr': learning_rate},
                                  {'params': tnet.parameters(), 'lr': learning_rate}
                                  ])

    for iter in range(start_epoch + 1, end_epoch + 1):
        losses = []
        t,A = tnet()
        t1 = t.view(s_b,s_c, s_I*s_J, s_i*s_j)
        tT = t1.permute(0,1,3,2)
        v = VNet(t1,tT)

        vs = v.shape
        ts = t.shape

        v1 = v.permute(0,1,4,2,3).view(s_b,s_c,s_t,s_i,s_j)

        vsnap_full = nn.functional.interpolate(v1.detach().cpu().squeeze(0), scale_factor=8, mode='nearest').numpy()
        tsnap_full = t.detach().cpu().numpy()

        t2 = t.view(s_b, s_c, s_I * s_J, s_i * s_j)
        v2 = v.view(s_b,s_c,s_i * s_j,s_t) / (s_i * s_j)
        tv = torch.matmul(t2, v2).view(s_b, s_c, s_I, s_J, s_t)

        optimizer.zero_grad()

        oemask_ = oemask.view(1, 1, s_I, s_J, 1)
        loss_fit_direct = 0.01 * criterion(oemask_ * tv,oemask_ * ZT)  # an L2 loss on the pixel-wise similarity between the predicted and observed videos, with overexposed pixels discarded

        loss_fit_dt = criterion1(oemask_ * tv,oemask_ * ZT)  # L1 loss on their difference, again with overexposed pixels ignored

        A = tnet.A.view(s_c, nvec, s_i, s_j)
        loss_mag = 0.0001 * torch.mean(A[:, 0, :, :].abs())

        smooth = 0.001
        Zmean_tm = Zmean.view(s_b, s_c, s_I, s_J, 1).expand(-1, -1, -1, -1, s_i * s_j)
        t_tm = t.view(s_b, s_c, s_I, s_J, s_i * s_j)
        if smooth > 0:
            t_tm2 = t_tm.view(s_b, s_c, s_I, s_J, s_i, s_j)
            # Smoothness loss on predicted T, on finite differences along both I and J
            # (on further thought the overexposure mask wouldn't be needed here, but let's keep it for consistency with the paper results)
            loss_smooth_dI = smooth * torch.mean(oemask_dI.view(1, 1, s_I - 1, s_J, 1, 1) * torch.abs(
                t_tm2[:, :, 1:, :, :, :] - t_tm2[:, :, :-1, :, :, :]))
            loss_smooth_dJ = smooth * torch.mean(oemask_dJ.view(1, 1, s_I, s_J - 1, 1, 1) * torch.abs(
                t_tm2[:, :, :, 1:, :, :] - t_tm2[:, :, :, :-1, :, :]))

        # Nonnegativity loss on T: discourage individual pixels that are below 0 by a large penalty
        loss_nonneg = 10 * torch.mean(torch.pow(torch.clamp(t, max=0), 2))

        # chromaticity loss: penalize the deviation of the T colors from their a![](ablation/data/light_transport/last_final/dots_target_0002469_disc.png)ve![](ablation/data/light_transport/last_final/dots_target_0002469_disc.png)rage
        t_Y = torch.mean(t, dim=1, keepdim=True)
        loss_T_chroma = 0.001 * criterion1(t, t_Y)

        #lossv = torch.sum(torch.abs(v)) * (1.4 * 1e-9) # if the dataset are disk
        lossv = torch.sum(torch.abs(v)) * (3.7 * 1e-12) # if the dataset are hands

        lossa = loss_fit_dt  + loss_fit_direct + loss_mag + loss_nonneg  + loss_smooth_dI + loss_smooth_dJ + loss_T_chroma
        loss = lossa + lossv

        loss.backward()
        optimizer.step()
        if iter % 1000 == 0:
            if iter % 10000 == 0:
                tva = tv.squeeze(0).detach().cpu().numpy()
                zta = ZT.squeeze(0).detach().cpu().numpy()
                ssim = SSIM(tva,zta,channel_axis=0)
                psnr = PSNR(tva,zta,data_range=1)
                print('iter %i' % (iter), lossa.item(),ssim.item(),psnr.item())
                with open(f'output_light_transport/lossLog-{logname}.txt', 'a+') as f:
                    f.write(f'iter: {iter}, loss: {lossa.item()},ssim:{ssim.item()},psnr:{psnr.item()}' + '\n')
            else:
                print('iter %i' % (iter), lossa.item())
                with open(f'output_light_transport/lossLog-{logname}.txt', 'a+') as f:
                    f.write(f'iter: {iter}, loss: {lossa.item()}' + '\n')

        if (iter) % 1000 == 0:
            print("Saving at iter: ", iter + 1)
            if not created_outputdir:
                # Create output folder with a hash and commit the current code to track it.
                created_outputdir = True
                hash_code = random.getrandbits(32)
                print(args.out_dir, machine_name, args.seq_name, args.dataset)
                outdir = args.out_dir + '/run_' + str(cur_time.strftime(
                    "%Y-%m-%d-%H%M--")) + "_" + machine_name + "_" + args.seq_name + "_" + args.dataset
                if args.skip_frames != 0:
                    outdir = outdir + "ff"
                outdir = outdir + '/'
                try:
                    os.mkdir(outdir)
                except OSError:
                    # TODO: was this an actual error condition, or just "directory already exists" catch?
                    None

            np.save(outdir + 'v.npy', v.detach().cpu().numpy())
            np.save(outdir + 't.npy', t.detach().cpu().numpy())
            torch.save(tnet, outdir + 'tnet.pth')
            torch.save(VNet, outdir + 'vnet.pth')

            def to8bit(x):
                return np.uint8(np.maximum(0.0, np.minimum(1.0, x)) * 255.)

            outdir_iter = os.path.join(outdir, 'vid_%08i' % iter)
            vsnap_full /= np.max(vsnap_full)
            vsnap_full = np.power(vsnap_full, 0.4545)  # gamma correction

            try:
                os.mkdir(outdir_iter)
            except OSError:
                None

            print("before saving check: ", s_t, np.min(gt_imgs), np.max(gt_imgs),
                  np.min(np.transpose(np.squeeze(vsnap_full[:, 0, :, :]), (1, 2, 0))),
                  np.max(np.transpose(np.squeeze(vsnap_full[:, 0, :, :]), (1, 2, 0))))
            print(np.hstack((to8bit(np.transpose(np.squeeze(vsnap_full[:, 0, :, :]), (1, 2, 0))), gt_imgs[0])).shape)
            tvs = np.squeeze(tv).permute(0, 3, 1, 2).reshape(s_c, s_t, s_I, s_J)

            for f in range(s_t):
                img_gen = np.hstack((to8bit(np.transpose(np.squeeze(vsnap_full[:, f, :, :]), (1, 2, 0))), gt_imgs[f]))
                img = img_gen
                com = np.squeeze(tvs[:, f, :, :]).permute(1, 2, 0).reshape(s_I, s_J, s_c).detach().cpu().numpy()

                imageio.imwrite(os.path.join(outdir_iter, 'f_%04d.png' % f), img)
                imageio.imwrite(os.path.join(outdir_iter, 'z_%04d.png' % f), com)

            # tsnap [1 c I J i j]
            tsnap_full /= np.max(tsnap_full)
            tsnap_full = np.power(tsnap_full, 0.4545)  # gamma correction![](output_light_transport/run_2022-10-26-0954--_yuqili3090_disc_dots/vid_00032000/f_0207.png)
            c = 0
            for i in range(s_i):
                for j in range(s_j):
                    # vsnap_full -= np.min(vsnap_full)
                    img_gen = np.transpose(np.squeeze(tsnap_full[0, :, :, :, i, j]), (1, 2, 0))
                    img = img_gen
                    imageio.imwrite(os.path.join(outdir_iter, 't_%06d.png' % c), to8bit(img))
                    c += 1



