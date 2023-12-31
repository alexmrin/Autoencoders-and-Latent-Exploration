import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import vars as v
import models
from data import *
import args
from utils import visualize


def vae_loss(preds, inputs, mean, logvar, MSE=True):
    difference_loss = F.mse_loss(preds, inputs, reduction='sum')
    KL_divergence = -.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
    return difference_loss + KL_divergence

def train():
    v.model.train()
    total_loss = 0.0
    t = tqdm(total=len(v.trainloader), desc=f"Epoch: {v.current_epoch}")
    for inputs, labels in v.trainloader:
        inputs = inputs.to(args.device)
        v.optimizer.zero_grad()
        preds, mean, logvar = v.model(inputs)
        loss = v.criterion(preds, inputs, mean, logvar)
        loss.backward()
        total_loss += loss.item()
        v.optimizer.step()

        t.update(1)
        t.set_postfix({"Loss": {loss.item()}})
    
    total_loss /= len(v.trainloader.dataset)
    v.writer.add_scalar("Training Loss", total_loss, v.current_epoch)

def test():
    v.model.eval()
    total_loss = 0.0
    t = tqdm(total=len(v.validloader), desc=f"Epoch: {v.current_epoch}")
    with torch.no_grad():
        for inputs, labels in v.validloader:
            inputs = inputs.to(args.device)
            preds, mean, logvar = v.model(inputs)
            loss = v.criterion(preds, inputs, mean, logvar)
            total_loss += loss.item()

            t.update(1)
            t.set_postfix({"Loss": {loss.item()}})

    total_loss /= len(v.validloader.dataset)
    v.lr_scheduler.step(total_loss)
    v.writer.add_scalar("Validation Loss", total_loss, v.current_epoch)

    sample = next(iter(v.validloader))
    input = sample[0][0].view(1, 1, 28, 28).to(args.device)
    visualize(v.model(input)[0], input)

def loop():
    v.model = v.model.to(args.device)
    #v.criterion = nn.MSELoss()
    v.criterion = vae_loss
    v.current_epoch = 1
    v.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad==True, v.model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    v.lr_scheduler = ReduceLROnPlateau(v.optimizer, mode='min', factor=0.1, patience=10)

    v.writer = SummaryWriter(log_dir=f"{args.save_path}")
    while v.current_epoch <= args.num_epochs:
        train()
        test()
        # saves model state
        torch.save(
            {
                "epoch": v.current_epoch,
                "model_state_dict": v.model.state_dict(),
                "optimizer_state_dict": v.optimizer.state_dict(),
            },
            f"{args.save_path}/model_weights.pt",
        )
        v.current_epoch += 1

if __name__ == "__main__":
    globals()[dataset_name]()
    v.model = getattr(models, model_name)()
    loop()