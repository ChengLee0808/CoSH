# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torchvision.utils import save_image
import os   
import scipy.io as sio
import imageio
import numpy as np
import data
import torch.nn.functional as F
import math
import time
from util.util import mkdir
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from torch.utils.data import DataLoader
from data.SH_dataset import SHDataset
from data.PSF_dataset import PSFDataset
# from data.PSF_dataset_sample import PSFDataset
from data.tif_dataset import TifDataset
from data.wfsless_dataset import WFSDataset

# 检查 output 和 label（gt）的取值范围
def check_tensor_range(tensor, name):
    actual_min = tensor.min().item()
    actual_max = tensor.max().item()
    print(f"{name}实际范围 [{actual_min:.4f}, {actual_max:.4f}]")


if __name__ == '__main__':
    opt = TestOptions().parse()
    # dataloader = data.create_dataloader(opt)
    # data_path =  '/disk/sdb2/PutYourFilesHere/lc/AOdata/20240127/flat/flat'
    # dataset = SHDataset(opt, data_path, train=False, split=0.9)

    # data_path = "/disk/sdb2/PutYourFilesHere/lc/far_field_data"
    # h5_folder = "/disk/sdb2/PutYourFilesHere/lc/far_field_data/h5_data_small"
    # mat_folder = '/disk/sdb2/PutYourFilesHere/lc/far_field_data/phase_maps_small'
    # dataset = PSFDataset(opt, data_path, h5_folder, mat_folder, train=False, split=0.9)

    # data_path = "/disk/sdb2/PutYourFilesHere/lc/far_field_data"
    # tif_folder = '/disk/sdb2/PutYourFilesHere/lc/far_field_data/0318_data/tif_data_1'
    # mat_folder = '/disk/sdb2/PutYourFilesHere/lc/far_field_data/0318_data/phase_maps_1'
    
    # dataset = TifDataset(opt, data_path, tif_folder, mat_folder,
    #                      train=False, split=0.95, num_samples=4900, seed=52, 
    #                      index_file='val_indices_0319.npy')
    
    data_path = '/disk/sdb2/PutYourFilesHere/lc/AOdata/20250404-PSF'
    dataset = WFSDataset(opt, data_path, train=False, split=0.9)

    dataset.initialize(opt)
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=(opt.phase == 'train'),
        num_workers=int(opt.nThreads),
        drop_last=(opt.phase != 'train')
        
    )
    model = Pix2PixModel(opt)
    
    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    else:
        model.to(opt.gpu_ids[0])
    model.eval()
    save_root = os.path.join(opt.checkpoints_dir, opt.name, 'test')
    mkdir(save_root)
    total_rms = 0
    total_num = 0
    rms = []
    start_time = time.time()

    for i, data_i in enumerate(dataloader):
        print('{} / {}'.format(i, len(dataloader)))
        if i * opt.batchSize >= opt.how_many:
            break
        imgs_num = data_i['label'].shape[0]
        start_time = time.time()
        out = model(data_i, mode='inference')
        end_time = time.time()
        execution_time = end_time - start_time
        print("程序执行时间： {:.2f} 秒".format(execution_time))
        if opt.save_per_img:
            try:
                for it in range(imgs_num):
                    save_name = os.path.join(save_root, '%04d_%04d.tiff' % (i, it))
                    save_image(out['fake_image'][it:it+1], save_name, padding=0, normalize=True)
            except OSError as err:
                print(err)
        else:
            label = data_i['label']
            output = out['fake_image'].cpu()
            output = output*50
            output = output * data_i['ref'].cpu()

            image = data_i['image'].cpu()*50
            # image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            image = image * data_i['ref'].cpu()

            label = ((label - torch.min(label)) / (torch.max(label) - torch.min(label)))
            imgs_ori = torch.cat((label.cpu(), output, image), 0)
            sio.savemat(os.path.join(save_root, '%04d.mat' % i), {'output': imgs_ori.numpy()})

            # 检查输出范围
            check_tensor_range(output, "output")
            check_tensor_range(image, "label")

            # 计算当前batch的MSE并累加
            batch_mse = F.mse_loss(output, image, reduction='sum').item()
            rms_err = torch.sqrt(torch.tensor(batch_mse / opt.batchSize / output.shape[3]/ output.shape[3]))/ (2*math.pi)
            print("RMS： {:.5f} lambda ".format(rms_err))
            rms.append(rms_err)
            total_rms = total_rms+rms_err

            # 显示并保存
            output = ((output - torch.min(output)) / (torch.max(output) - torch.min(output)))
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            imgs = torch.cat((label.cpu(), output, image), 0)
            try:
                save_name = os.path.join(save_root, '%04d.tiff' % i)
                save_image(imgs, save_name, nrow=imgs_num, padding=0, normalize=True)
            except OSError as err:
                print(err)

    # sio.savemat(os.path.join(save_root, 'rms_val.mat'), {'rms': rms})
    # 计算整个验证数据集上的RMS
    total_rms = total_rms / len(dataloader)
    
    print("RMS_avgall: {:.5f} lambda ".format(total_rms))
