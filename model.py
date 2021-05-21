import torch
import torch.nn as nn
import torchvision.models as models
from dataset import *


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image,itos, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if itos[predicted.item()] == "<EOS>":
                    break

        return  [itos[idx] for idx in result_caption]



if __name__ == '__main__':
    model = CNNtoRNN(config.embed_size, config.hidden_size, config.vocab_size, config.num_layers).to(config.device)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),]
    )
    img_names, captions = make_data()

    dataset = Dataset(img_names, captions, transform=transform)

    pad_idx = dataset.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers ,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    imgs, captions = next(iter(loader))
    outputs = model(imgs, captions[:-1])
    print(outputs.shape) # torch.Size([21, 8, 2595])
