from numpy.random.mtrand import sample
from pandas.core.reshape.reshape import stack_multiple
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.distributed as dist
import torch.nn.functional as F
from medpy import metric
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 

from model import VNet
from dataloader import LAHeart
from dataloader import RandomCrop, RandomNoise, RandomRotFlip, ToTensor, CreateOnehotLabel

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model_a = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
# save_mode_path = os.path.join('iter_6000.pth')
# model_a.load_state_dict(torch.load(save_mode_path))
model_a = model_a.to(device)
model_b = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
model_b = model_b.to(device)

# Metrics
loss_seg = nn.CrossEntropyLoss()
loss_cps = nn.CrossEntropyLoss()

# Get hardlabel
gethardlabel = CreateOnehotLabel()

# Dataloader
batch_size = 1
num_workers = 4
train_transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop((112, 112, 80)),
                          ToTensor(),
                          ])
trainset = LAHeart(split='labelled_train', transform=train_transform, num=16)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
unlabelled_train_transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop((112, 112, 80)),
                          ToTensor(),
                          ])
unlabelled_trainset = LAHeart(split='unlabelled_train', transform=unlabelled_train_transform, num=64)
unlabelled_trainloader = torch.utils.data.DataLoader(unlabelled_trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)

test_transform = transforms.Compose([
                          RandomCrop((112, 112, 80)),
                          ToTensor(),
                          ])
testset = LAHeart(split='test', transform=test_transform, num=16)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)



# Optimizers
learn_rate = 0.01
optimizer_a = optim.SGD(model_a.parameters(), lr=learn_rate, momentum=0.9, weight_decay=0.0005)
optimizer_b = optim.SGD(model_b.parameters(), lr=learn_rate, momentum=0.9, weight_decay=0.0005)

# Tensorboard init
writer = SummaryWriter()

# TRAINING
Max_epoch = 800
model_a.train()
model_b.train()            

for epoch in range(Max_epoch):
    print(f'Epoch {epoch+1}/{Max_epoch}')
    print('-' * 20)

    # Supervised Training
    # Labelled Dataset
    sup_loss_cps = 0.0
    sup_loss_seg_a = 0.0
    sup_loss_seg_b = 0.0
    sup_loss = 0.0

    for batch_idx, sample in tqdm(enumerate(trainloader)):
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        images = sample["image"]
        labels = sample["label"]
        # im = sample["image"]
        # la = sample["label"]
        images = images.to(device)
        labels = labels.to(device)
        outputs_a = model_a(images)
        #outputs_b = model_b(images)
        # hardlabel_a = gethardlabel(outputs_a).to(device)
        # hardlabel_b = gethardlabel(outputs_b).to(device)

        # _, hardlabel_a = torch.max(outputs_a, dim=1)
        # _, hardlabel_b = torch.max(outputs_b, dim=1)
        # hardlabel_a.type(torch.float32)
        # hardlabel_b.type(torch.float32)
        
        # cps_loss = loss_cps(outputs_a, hardlabel_b) + loss_cps(outputs_b, hardlabel_a)
        seg_loss_a = loss_seg(outputs_a, labels)
        #seg_loss_b = loss_seg(outputs_b, labels)
        # dist.all_reduce(seg_loss_a, dist.ReduceOp.SUM)
        # loss = seg_loss_a + seg_loss_b + 0.5 * cps_loss
        # loss.backward()
        loss = seg_loss_a #+ seg_loss_b
        loss.backward()

        optimizer_a.step()
        # optimizer_b.step()
        # sup_loss_cps += cps_loss.item()
        sup_loss_seg_a += seg_loss_a.item()
        #sup_loss_seg_b += seg_loss_b.item()
        sup_loss += loss.item()
    
    print(sup_loss_cps/16)
    print(sup_loss_seg_a/16)
    print(sup_loss_seg_b/16)
    # Save Metrics
    # sup_loss_cps = sup_loss_cps/len(trainset)
    # sup_loss_seg_a = sup_loss_seg_a/len(trainset)
    # sup_loss_seg_b = sup_loss_seg_b/len(trainset)
    # writer.add_scalar("CPS Loss/Labelled", sup_loss_cps, epoch)
    # writer.add_scalar("SegLoss Model A/Labelled", sup_loss_seg_a, epoch)
    # writer.add_scalar("SegLoss Model B/Labelled", sup_loss_seg_b, epoch)

    # Unsupervised Training
    # Unlabelled Dataset
    # unsup_loss_cps = 0.0

    # for batch_idx, sample in tqdm(enumerate(unlabelled_trainloader)):
    #     images = sample["image"]
    #     labels = sample["label"]
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     outputs_a = model_a(images)
    #     outputs_b = model_b(images)
    #     # hardlabel_a = gethardlabel(outputs_a).to(device)
    #     # hardlabel_b = gethardlabel(outputs_b).to(device)

    #     _, hardlabel_a = torch.max(outputs_a, dim=1)
    #     _, hardlabel_b = torch.max(outputs_b, dim=1)
    #     hardlabel_a.type(torch.float32)
    #     hardlabel_b.type(torch.float32)

    #     unsup_cps_loss = loss_cps(outputs_a, hardlabel_b) + loss_cps(outputs_b, hardlabel_a)

    #     unsup_cps_loss.backward()
    #     optimizer_a.step()
    #     optimizer_b.step()

    #     unsup_loss_cps += unsup_cps_loss.item()
    #     optimizer_a.zero_grad()
    #     optimizer_b.zero_grad()

    # # Save Metrics
    # unsup_loss_cps = unsup_loss_cps/len(unlabelled_trainset)
    # writer.add_scalar("CPS Loss/Unlabelled", unsup_loss_cps, epoch)

    # # Checkpoints
    print("=> Epoch:{} - sup_loss: {:.4f}".format(epoch+1, sup_loss/len(trainset)))
    # print("=> Epoch:{} - sup_cpsloss: {:.4f}".format(epoch+1, sup_loss_cps))
    # print("=> Epoch:{} - unsup_loss: {:.4f}".format(epoch+1, unsup_loss_cps))

    this_epoch = epoch + 1
    if this_epoch % 3 == 0:
        torch.save(model_a.state_dict(), "am.pth")
        torch.save(model_b.state_dict(), "bm.pth")
        torch.save(optimizer_a.state_dict(), "optim_a.pth")
        torch.save(optimizer_b.state_dict(), "optim_b.pth")

        # Testing part if thisepoch%50 == 0
    
    if epoch != 3:
        model_a.eval()
        # model_b.eval()
        dice = 0.0
        asd= 0.0
        jc = 0.0
        hd = 0.0
        for batch_idx, sample in tqdm(enumerate(testloader)):

            images = sample["image"]
            labels = sample["label"]
            images = images.to(device)
            # labels = labels.to(device)
            # outputs = model_a(images)

            # outputs_a = outputs.detach().cpu().numpy()
            labels = labels.numpy()

            # outputs_a = outputs_a[0][1]
            # outputs_a = np.expand_dims(outputs_a, 0)

            outputs = model_a(images)
            # im = images.cpu().numpy()
            outsoft = F.softmax(outputs, dim=1)
            outsoft = outsoft.detach().cpu().numpy()
            # outsoft = outsoft.cpu().data.numpy()
            outsoft = outsoft[0,:,:,:,:]
            # score_map = np.zeros((2, ) + outsoft[0].shape).astype(np.float32)
            label_map = np.argmax(outsoft, axis = 0)

            dice += metric.binary.dc(label_map, labels[:][0])
            jc += metric.binary.jc(label_map, labels[:][0])
            asd += metric.binary.asd(label_map, labels[:][0])
            hd += metric.binary.hd95(label_map, labels[:][0])
        
        print(dice)
        print(jc)
        print(hd)
        print(asd)
#     dice /= len(path_list)
# jc /= len(path_list)
# asd /= len(path_list)
# hd /= len(path_list)

writer.flush()
writer.close()




# outputs = model_a(images)
# im = images.cpu().numpy()
# outsoft = F.softmax(outputs_a, dim=1)
# outsoft = outsoft.detach().cpu().numpy()
# # outsoft = outsoft.cpu().data.numpy()
# outsoft = outsoft[0,:,:,:,:]
# # score_map = np.zeros((2, ) + outsoft[0].shape).astype(np.float32)
# label_map = np.argmax(outsoft, axis = 0)

# labels = labels.numpy()
# print(labels)

# print(labels.shape)
# print(outputs_a.shape)
# outputs_a = outputs_a[0][1]
# outputs_a = np.expand_dims(outputs_a, 0)

# y1 = net(test_patch)
# y = F.softmax(y1, dim=1)
#                 y = y.cpu().data.numpy()
#                 y = y[0,:,:,:,:]
#                 score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
#                   = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
#                 cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
#                   = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
#     score_map = score_map/np.expand_dims(cnt,axis=0)
#     label_map = np.argmax(score_map, axis = 0)

