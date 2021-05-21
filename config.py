model_save_path = './weights/model.pth'
img_folder = './data/flickr8k/images'
caption_path = './data/flickr8k/captions.txt'
data = './data/data.json'
itos = './data/itos.json'
stoi = './data/stoi.json'
freq_threshold = 5
batch_size = 8
num_workers = 4


import torch
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = False
train_CNN = False

# Hyperparameters
embed_size = 256
hidden_size = 256
vocab_size = 2595 # len(stoi)
num_layers = 1
learning_rate = 3e-4
num_epochs = 100
