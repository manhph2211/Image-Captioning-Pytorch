import config
from tqdm import tqdm


def train_fn(model, data_loader,criterion,optimizer):
    model.train()
    device = config.device
    epoch_loss = 0
    for imgs, captions in tqdm(data_loader):
        imgs = imgs.to(device)
        captions = captions.to(device)
        outputs = model(imgs, captions[:-1])
        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        epoch_loss+=loss.item()
        optimizer.step()
    return epoch_loss


def val_fn(model, data_loader,criterion):
    model.eval()
    device = config.device
    epoch_loss = 0
    for imgs, captions in tqdm(data_loader):
        imgs = imgs.to(device)
        captions = captions.to(device)
        outputs = model(imgs, captions[:-1])
        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
        )
        epoch_loss+=loss.item()
    return epoch_loss