from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utils import *
import config
import torch
import cv2


class IMDataset(Dataset):
    def __init__(self, img_names, captions, transform=None):
        self.img_names = img_names
        self.captions = captions
        self.transform = transform
        self.itos = read_json(config.itos)
        self.stoi = read_json(config.stoi)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_name = self.img_names[index]
        img = cv2.imread(os.path.join(config.img_folder, img_name))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = torch.FloatTensor(img/255)
        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.stoi["<SOS>"]]
        numericalized_caption += numericalize(caption, self.stoi)
        numericalized_caption.append(self.stoi["<EOS>"])
        caplen = len(numericalized_caption) + 2
        return img, torch.tensor(numericalized_caption),torch.LongTensor(caplen)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        imgs = torch.cat(imgs, dim=0)
        caplens = [item[2] for item in batch]
        caplens = torch.cat(caplens,dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return imgs, targets, caplens


if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.Resize((224, 224))]# transforms.ToTensor(), ]
    )

    dataset = IMDataset(img_names_test,captions_test, transform=transform)

    pad_idx = dataset.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    for idx, (imgs, captions,_) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)