from utils import *
from dataset import MyCollate, IMDataset
from model import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import config
from engine import *


if __name__ == "__main__":
    
    device = config.device
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            normalize,
        ]
    )
    stoi = read_json(config.stoi)
    pad_idx = stoi["<PAD>"]
    train_dataset = IMDataset(img_names_train, captions_train, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    val_dataset = IMDataset(img_names_val, captions_val, transform=transform)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    test_dataset = IMDataset(img_names_test, captions_test, transform=transform)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )


    decoder = DecoderWithAttention(attention_dim=config.attention_dim,
                                       embed_dim=config.emb_dim,
                                       decoder_dim=config.decoder_dim,
                                       vocab_size=len(stoi),
                                       dropout= config.dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=config.decoder_lr)
    encoder = Encoder()
    fine_tune_encoder = False
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=config.encoder_lr) if fine_tune_encoder else None


    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    best_loss_val = 9999
    log = []

    for epoch in range(config.num_epochs + 1):
        train_loss = train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer)
        val_loss = validate(val_loader, encoder, decoder, criterion)
        log_epoch = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        log.append(log_epoch)
        df = pd.DataFrame(log)
        df.to_csv(config.save_log)
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer)
        print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f} ".format(epoch + 1, train_loss, val_loss))



