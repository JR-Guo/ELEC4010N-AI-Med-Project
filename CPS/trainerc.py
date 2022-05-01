from numpy.random.mtrand import sample
from pandas.core.reshape.reshape import stack_multiple
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import transforms
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
        
        cps_loss = loss_cps(outputs_a, hardlabel_b) + loss_cps(outputs_b, hardlabel_a)
        seg_loss_a = loss_seg(outputs_a, labels)
        seg_loss_b = loss_seg(outputs_b, labels)
        
        loss = seg_loss_a + seg_loss_b + cps_loss
        loss.backward()
        optimizer_a.step()
        optimizer_b.step()
    
        sup_loss_cps += cps_loss.item()
        sup_loss_seg_a += seg_loss_a.item()
        sup_loss_seg_b += seg_loss_b.item()
        sup_loss += loss.item()
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
    
    # Save Metrics
    sup_loss_cps = sup_loss_cps/len(trainset)
    sup_loss_seg_a = sup_loss_seg_a/len(trainset)
    sup_loss_seg_b = sup_loss_seg_b/len(trainset)
    writer.add_scalar("CPS Loss/Labelled", sup_loss_cps, epoch)
    writer.add_scalar("SegLoss Model A/Labelled", sup_loss_seg_a, epoch)
    writer.add_scalar("SegLoss Model B/Labelled", sup_loss_seg_b, epoch)

    # Unsupervised Training
    # Unlabelled Dataset
    unsup_loss_cps = 0.0

    for batch_idx, sample in tqdm(enumerate(unlabelled_trainloader)):

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

        unsup_cps_loss = loss_cps(outputs_a, hardlabel_b) + loss_cps(outputs_b, hardlabel_a)

        unsup_cps_loss.backward()
        optimizer_a.step()
        optimizer_b.step()

        unsup_loss_cps += unsup_cps_loss.item()
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()

    # Save Metrics
    unsup_loss_cps = unsup_loss_cps/len(unlabelled_trainset)
    writer.add_scalar("CPS Loss/Unlabelled", unsup_loss_cps, epoch)

    # Checkpoints
    print("=> Epoch:{} - sup_loss: {:.4f}".format(epoch+1, sup_loss/len(trainset)))
    print("=> Epoch:{} - sup_cpsloss: {:.4f}".format(epoch+1, sup_loss_cps))
    print("=> Epoch:{} - unsup_loss: {:.4f}".format(epoch+1, unsup_loss_cps))

    this_epoch = epoch + 1
    if this_epoch % 25 == 0:
        torch.save(model_a.state_dict(), "model_a.pth")
        torch.save(model_b.state_dict(), "model_b.pth")
        torch.save(optimizer_a.state_dict(), "optim_a.pth")
        torch.save(optimizer_b.state_dict(), "optim_b.pth")

writer.flush()
writer.close()
