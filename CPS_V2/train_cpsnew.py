from numpy.random.mtrand import sample
# from pandas.core.reshape.reshape import stack_multiple
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
import PIL
import numpy as np
from PIL import Image
from model import VNet
from dataset import LAHeart
# from dataloader import LAHeart
from dataloader import RandomCrop, RandomNoise, RandomRotFlip, ToTensor, CreateOnehotLabel

def dice_loss(sco, target, acc=0):
    target = target.float()
    score = F.softmax(sco, dim=1)
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    if acc == 1:
        return loss
    loss = 1 - loss
    return loss

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model_a = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).cuda()
# model_a = model_a.to(device)
model_b = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).cuda()
# model_b = model_b.to(device)

# Dataloader
batch_size = 2
num_workers = 4
train_transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop((112, 112, 80)),
                          ToTensor(),
                          ])
# trainset = LAHeart(split='labelled_train', transform=train_transform, num=16)
trainset = LAHeart(
        split='Training Set', 
        label= True,
        transform=train_transform
        )

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)
unlabelled_train_transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop((112, 112, 80)),
                          ToTensor(),
                          ])
# unlabelled_trainset = LAHeart(split='unlabelled_train', transform=unlabelled_train_transform, num=64)
unlabelled_trainset = LAHeart(
        split='Training Set', 
        label= False,
        transform=train_transform
        )
unlabelled_trainloader = torch.utils.data.DataLoader(unlabelled_trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers)

# Optimizers
Max_epoch = 800
learn_rate = 0.01
optimizer_a = optim.SGD(model_a.parameters(), lr=learn_rate, momentum=0.9, weight_decay=0.0005)
optimizer_b = optim.SGD(model_b.parameters(), lr=learn_rate, momentum=0.9, weight_decay=0.0005)

lr_scheduler_a = optim.lr_scheduler.StepLR(
        optimizer_a, 
        step_size=1, 
        gamma=np.power(0.001, 1 / Max_epoch)
    )
lr_scheduler_b = optim.lr_scheduler.StepLR(
        optimizer_a, 
        step_size=1, 
        gamma=np.power(0.001, 1 / Max_epoch)
    )

# Tensorboard init
writer = SummaryWriter()

# TRAINING            

for epoch in range(Max_epoch):
    print(f'Epoch {epoch+1}/{Max_epoch}')
    print('-' * 30)
    model_a.train()
    model_b.train()            

    # Supervised Training
    # Labelled Dataset
    sup_loss_cps = 0.0
    sup_loss_seg_a = 0.0
    # sup_loss_seg_b = 0.0
    sup_loss = 0.0
    dice_acc_a = 0.0

    for batch_idx, sample in tqdm(enumerate(trainloader)):
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        images = sample["image"]
        # print(images.shape)
        labels = sample["label"]
        images = images.to(device)
        labels = labels.to(device)
        outputs_a = model_a(images)
        # print(outputs_a.shape, images.shape, labels.shape)
        outputs_b = model_b(images)
        # out_scores_a = F.softmax(outputs_a, dim=1)
        # if batch_idx == 0:
        #         print(outputs_a.shape, labels.shape)
        #         print(1-dice_loss(out_scores[:, 1, ...], labels == 1))
        # hardlabel_a = gethardlabel(outputs_a).to(device)
        # hardlabel_b = gethardlabel(outputs_b).to(device)

        _, hardlabel_a = torch.max(outputs_a, dim=1)
        _, hardlabel_b = torch.max(outputs_b, dim=1)
        
        hardlabel_a.type(torch.float32)
        hardlabel_b.type(torch.float32)
        # print(hardlabel_a.shape, hardlabel_b.shape)
        cps_loss = (F.cross_entropy(outputs_a, hardlabel_b) + F.cross_entropy(outputs_b, hardlabel_a))
        seg_loss_a = F.cross_entropy(outputs_a, labels)
        seg_loss_b = F.cross_entropy(outputs_b, labels)
        
        # Compute dice per epoch
        dice_a = dice_loss(outputs_a, labels, 1)
        dice_acc_a += dice_a

        loss = seg_loss_a + seg_loss_b + 0.05 * cps_loss
        loss.backward()
        optimizer_a.step()
        optimizer_b.step()
        # print("LABELLED")
        # print(cps_loss.item())
        # print(seg_loss_a.item())
        # print(loss.item())
        sup_loss_cps += cps_loss.item()
        sup_loss_seg_a += seg_loss_a.item()
        # sup_loss_seg_b += seg_loss_b.item()
        sup_loss += loss.item()
        
    # Save Metrics
    sup_loss_cps = sup_loss_cps/len(trainset)
    sup_loss_seg_a = sup_loss_seg_a/len(trainset)
    # sup_loss_seg_b = sup_loss_seg_b/len(trainset)
    dice_acc_a = dice_acc_a/len(trainset)
    print("labelled dice {}".format(dice_acc_a))
    sup_loss /= len(trainset)
    print("labelled loss {}".format(sup_loss))
    writer.add_scalar("CPS Loss/Labelled", sup_loss_cps, epoch)
    writer.add_scalar("SegLoss Model A/Labelled", sup_loss_seg_a, epoch)
    # writer.add_scalar("SegLoss Model B/Labelled", sup_loss_seg_b, epoch)
    writer.add_scalar("Dice Accuracy/ Labelled", dice_acc_a, epoch)

#     # Unsupervised Training
#     # Unlabelled Dataset
    unsup_loss_cps = 0.0
    dice_acc_a = 0.0

    if epoch > 48: #80
        for batch_idx, sample in tqdm(enumerate(unlabelled_trainloader)):
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            images = sample["image"]
            labels = sample["label"]
            images = images.to(device)
            labels = labels.to(device)
            outputs_a = model_a(images)
            outputs_b = model_b(images)
            # hardlabel_a = gethardlabel(outputs_a).to(device)
            # hardlabel_b = gethardlabel(outputs_b).to(device)

            _, hardlabel_a = torch.max(outputs_a, dim=1)
            _, hardlabel_b = torch.max(outputs_b, dim=1)
            hardlabel_a.type(torch.float32)
            hardlabel_b.type(torch.float32)

            unsup_cps_loss = 0.01*(F.cross_entropy(outputs_a, hardlabel_b) + F.cross_entropy(outputs_b, hardlabel_a))
            # print("UNLABELLED")
            # print(unsup_cps_loss.item())
            unsup_cps_loss.backward()
            optimizer_a.step()
            optimizer_b.step()
            # Compute dice per epoch
            dice_a = dice_loss(outputs_a, labels, 1)
            dice_acc_a += dice_a
            unsup_loss_cps += unsup_cps_loss.item()

    # Save Metrics
    unsup_loss_cps = unsup_loss_cps/len(unlabelled_trainset)
    dice_acc_a /= len(unlabelled_trainset)
    print("unlabelled dice {}".format(dice_acc_a))
    writer.add_scalar("CPS Loss/Unlabelled", unsup_loss_cps, epoch)
    writer.add_scalar("Dice Accuracy/Unlabelled", dice_acc_a, epoch)

    # Checkpoints
    # print("=> Epoch:{} - sup_loss: {:.4f}".format(epoch+1, sup_loss/len(trainset)))
    # print("=> Epoch:{} - sup_cpsloss: {:.4f}".format(epoch+1, sup_loss_cps))
    # print("=> Epoch:{} - unsup_loss: {:.4f}".format(epoch+1, unsup_loss_cps))

    this_epoch = epoch + 1
    if this_epoch % 20 == 0:
        print("save model!!!")
        torch.save(model_a.state_dict(), "model_a_lb_ulb.pth")
        torch.save(model_b.state_dict(), "model_b_lb_ulb.pth")
        torch.save(optimizer_a.state_dict(), "optim_a.pth")
        torch.save(optimizer_b.state_dict(), "optim_b.pth")

    if this_epoch % 40 == 0:
        image3d = images.detach().cpu().numpy()
        label3d = labels.detach().cpu().numpy()
        pred3d = hardlabel_a.detach().cpu().numpy()
        # print(image3d.shape)
        # print(label3d.shape)
        # print(pred3d.shape)

        for i in range(3):
            imageslice = image3d[0][0][:, :, i*20]
            labelslice = label3d[0][:, :, i*20]
            predslice = pred3d[0][:, :, i*20]
            for index in range(112):
                for j in range(112):
                    # Can use for testing visualisations
                    # temp[index][j] *= 25.5
                    # templa[index][j] *= 25.5
                    if imageslice[index][j] != 0:
                        imageslice[index][j] *= 15
                    if labelslice[index][j] != 0:
                        labelslice[index][j] = 255
                    if predslice[index][j] != 0:
                        predslice[index][j] = 255
                                  
            # print(imageslice.shape)
            # print(predslice.shape)
            im = Image.fromarray(np.int8(imageslice)).convert('L')
            impath = "./image/" + str(this_epoch) + "_" + str(i) + ".png"
            im.save(impath)
            
            pr = Image.fromarray(np.int8(predslice)).convert('L')
            prpath = "./pred/" + str(this_epoch) + "_" + str(i) + ".png"
            pr.save(prpath)

            la = Image.fromarray(np.int8(labelslice)).convert('L')
            lapath = "./label/" + str(this_epoch) + "_" + str(i) + ".png"
            la.save(lapath)

    lr_scheduler_a.step()
    lr_scheduler_b.step()

writer.flush()
writer.close()