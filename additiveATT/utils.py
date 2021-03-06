import glob
import os
import config
import json
from sklearn import model_selection
import spacy  # for tokenizer
import torch
spacy_eng = spacy.load("en_core_web_sm")


def make_data(txt_path = config.caption_path):
    img_names = []
    captions = []
    with open(txt_path,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.split(',')
            img_names.append(line[0])
            captions.append(','.join(line[1:])[:-1])

    return img_names,captions


def split_data(img_paths,targets):
    img_names_train, img_names_test, captions_train, captions_test = model_selection.train_test_split(img_paths, targets, test_size=0.05, random_state=1)
    img_names_train, img_names_val, captions_train, captions_val = model_selection.train_test_split(img_names_train, captions_train, test_size=0.2, random_state=1) # 0.25 x 0.8 = 0.2
    return img_names_train,captions_train,img_names_val,captions_val,img_names_test,captions_test


def write_json(data,data_path):
    with open(data_path,'w') as f:
        json.dump(data,f,indent=4)


def read_json(data_path):
    with open(data_path,'r') as f:
        data = json.load(f)
    return data


class Vocab:

    def __init__(self,captions,freq_threshold):
        self.captions = captions
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def get_vocab(self):

        word2fre ={}
        idx = 4 # >3!
        self.max_length = 0
        for caption in self.captions:
            for i,word in enumerate(tokenizer_eng(caption)):
                if word not in word2fre:
                    word2fre[word] = 1
                else:
                    word2fre[word]+=1
                if word2fre[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

            if i + 1 > self.max_length:
                self.max_length = i + 1


def tokenizer_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]


def numericalize(text,stoi):
    tokenized_text = tokenizer_eng(text)

    return [
        stoi[token] if token in stoi else stoi["<UNK>"]
        for token in tokenized_text
    ]


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer):
  
    state = {'epoch': epoch,
            
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = config.model_save_path + '.tar'
    torch.save(state, filename)
   

# img_paths = glob.glob(os.path.join(config.img_folder,'*.jpg'))
# print(len(img_paths))  # 8091 images
img_names,captions = make_data()
img_names_train,captions_train,img_names_val,captions_val,img_names_test,captions_test = split_data(img_names,captions)
# vocab = Vocab(captions_train,config.freq_threshold)
# vocab.get_vocab()
# write_json(vocab.itos,config.itos)
# write_json(vocab.stoi,config.stoi)
# print(vocab.max_length) # 42s
stoi = read_json(config.stoi)
#print(len(stoi))
#print(numericalize(' hello girl',stoi)) #[3, 2498 , 24]
