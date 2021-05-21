import torch
import torch.nn as nn
from engine import train_fn,val_fn
from utils import *
import pandas as pd
import torchvision.transforms as transforms
from dataset import IMDataset, MyCollate
from torch.utils.data import DataLoader
from model import CNNtoRNN
import torch.optim as optim


if __name__ == "__main__":

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    stoi = read_json(config.stoi)
    pad_idx = stoi["<PAD>"]
    img_names, captions = make_data()
    img_names_train,captions_train,img_names_val,captions_val,img_names_test,captions_test = split_data(img_names,captions)

    train_dataset = IMDataset(img_names_train, captions_train, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers ,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    val_dataset = IMDataset(img_names_val, captions_val, transform=transform)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers ,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    test_dataset = IMDataset(img_names_test, captions_test, transform=transform)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers ,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    model = CNNtoRNN(config.embed_size, config.hidden_size, config.vocab_size, config.num_layers).to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss_val = 9999
    log=[]
    
    for epoch in range(config.num_epochs+1):
        train_loss = train_fn(model, train_loader,criterion,optimizer)
        val_loss = val_fn(model, val_loader,criterion)
        log_epoch = {"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss}
        log.append(log_epoch)
        df = pd.DataFrame(log)
        df.to_csv(config.save_log) 
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            torch.save(model.state_dict(),config.model_save_path)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f} ".format(epoch + 1,train_loss, val_loss))




