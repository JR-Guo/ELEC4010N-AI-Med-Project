from dataset import LAHeart
from transforms import RandomCrop, RandomRotFlip, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from vnet import VNet
import numpy as np
import torch
import torch.nn.functional as F
import os


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


if __name__ == '__main__':
    max_epoch = 1000
    batch_size = 2
    patch_size = (112, 112, 80)
    num_classes = 2
    # torch.nn.parallel()

    model =  VNet(n_channels=1, n_classes=num_classes,\
         normalization='batchnorm', has_dropout=True).cuda()
    # model = torch.nn.Module(model)
    # model = torch.nn.DataParallel(model)
    train_dst = LAHeart(
        split='Training Set', 
        label= True,
        transform=transforms.Compose([
            RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ])
    )
    print(len(train_dst))
    train_loader = DataLoader(
        train_dst, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2, 
        pin_memory=True
    )

    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.01, 
        momentum=0.9, 
        weight_decay=1e-4
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=np.power(0.001, 1 / max_epoch)
    )

    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model)
    
    for epoch in range(max_epoch + 1):
        model.train()
        loss_list = []
        for batch in train_loader:
            # continue
            image, label = batch['image'].cuda(), batch['label'].cuda()
            out = model(image)
            out_scores = F.softmax(out, dim=1)

            loss_ce = F.cross_entropy(out, label)
            loss_dice = dice_loss(out_scores[:, 1, ...], label == 1)
            loss = (loss_ce + loss_dice)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        
        lr_scheduler.step()
        print(f'epoch {epoch}, loss {np.mean(loss_list)}')

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), 
                f'./logs/ep_fullset_{epoch}.pth'
            )
