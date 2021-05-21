from model import CNNtoRNN
from dataset import IMDataset,MyCollate
import config
import matplotlib.pyplot as plt 
import cv2
from utils import read_json, make_data,split_data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch


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

    # train_dataset = IMDataset(img_names_train, captions_train, transform=transform)

    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers ,
    #     collate_fn=MyCollate(pad_idx=pad_idx),
    # )

    # val_dataset = IMDataset(img_names_val, captions_val, transform=transform)

    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers ,
    #     collate_fn=MyCollate(pad_idx=pad_idx),
    # )

    test_dataset = IMDataset(img_names_test, captions_test, transform=transform)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers ,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    model = CNNtoRNN(config.embed_size, config.hidden_size, config.vocab_size, config.num_layers).to(config.device)
    model.load_state_dict(torch.load(config.model_save_path,map_location=torch.device('cpu')))
    model.eval()

    imgs, captions = next(iter(test_loader))
    itos = read_json(config.itos)
    img,caption = next(iter(zip(imgs,captions.T)))
    caption = [itos[str(x)] for x in caption.numpy() if x not in (0,1,2,3)]
    print(' '.join(caption))
    out = model.caption_image(img.reshape(1,img.shape[0],img.shape[1],img.shape[2]),itos)
    plt.title(' '.join(out))
    plt.imshow((imgs[0].permute(1,2,0)*0.5+0.5))
    plt.show()
        
    

