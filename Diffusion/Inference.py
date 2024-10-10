import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from Diffusion import GaussianDDPMSampler, GaussianDiffusionTrainer,GaussianDDIMSampler
from Diffusion.Model import UNet
from .Scheduler import GradualWarmupScheduler


def infer(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        geneImage = []

        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        sampler = GaussianDDPMSampler(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        

        for i in range (100):

            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], 3, 64, 64], device=device)

            # 进行采样
            sampledImgs = sampler(noisyImage)
            # 采样的归一化
            sampledImgs = sampledImgs * 0.5 + 0.5  

            geneImage.append(sampledImgs)

            print("完成了第  ", i , "  轮图片生成！")


        for i in range(len(geneImage)):
            for j in range (geneImage[0].shape[0]):
                save_image(geneImage[i][j], os.path.join(modelConfig["sampled_dir"],  'image_'+ str(i * geneImage[0].shape[0] + j) + '.png'), nrow=modelConfig["nrow"])
