import torch
import config
import tqdm


def train_fn(model, data_loader,criterion,optimizer):
    model.train()
    device = config.device
    epoch_loss = 0
    for idx, (imgs, captions) in tqdm(enumerate(data_loader)):
        imgs = imgs.to(device)
        captions = captions.to(device)
        outputs = model(imgs, captions[:-1])
        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward(loss)
        epoch_loss+=loss.item()
        optimizer.step()
    return epoch_loss


def eval_fn(model, data_loader,criterion):
    model.eval()
    device = config.device
    epoch_loss = 0
    for idx, (imgs, captions) in tqdm(enumerate(data_loader)):
        imgs = imgs.to(device)
        captions = captions.to(device)
        outputs = model(imgs, captions[:-1])
        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
        )
        epoch_loss+=loss.item()
    return epoch_loss