import torch

model_save_path = './weights/model.pth'
save_log = "./weights/log.csv"
img_folder = '../data/flickr8k/images'
caption_path = '../data/flickr8k/captions.txt'
data = '../data/data.json'
itos = '../data/itos.json'
stoi = '../data/stoi.json'

freq_threshold = 3
batch_size = 4
num_workers = 4

torch.backends.cudnn.benchmark = True
device = 'cuda'

grad_clip = 5
alpha_c = 1
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder

max_len = 45

num_epochs = 1

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5