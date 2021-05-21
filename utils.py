import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import glob
import os 
import config
import json 
import spacy  # for tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd 


def make_data(txt_path = config.caption_path):
	data = {}
	with open(txt_path,'r') as f:
		lines = f.readlines()
		for line in lines[1:]:
			line = line.split(',')
			img_name = line[0]
			caption = ','.join(line[1:])[:-1]
			if img_name not in data:
				data[img_name] = [caption]
			else:
				data[img_name].append(caption)

	write_json(data,config.data)


def split_data(all_data_json_path=config.data, ratio=[0.8, 0.15, 0.05]):
    data = read_json(all_data_json_path)

    all_image_paths = []
    all_captions = []
    for image_path, caption_path in data.items():
        all_image_paths.append(image_path)
        all_captions.append(caption_path)

    normalized_ratio = [e / sum(ratio) for e in ratio]
    train_image_paths, vt_image_paths, \
    train_captions, vt_captions = train_test_split(all_image_paths, all_captions,
                                                       test_size=1 - normalized_ratio[0])
    val_image_paths, test_image_paths, \
    val_captions, test_captions = train_test_split(vt_image_paths, vt_captions,
                                                       test_size=normalized_ratio[-1] / (1 - normalized_ratio[0]))
    def save_data(image_paths, captions, save_path):
        data_dict = {}
        for image_path, caption_path in zip(image_paths, captions):
            data_dict[image_path] = caption_path
        write_json(data_dict,save_path)

    save_data(train_image_paths, train_captions, config.train_data)
    save_data(val_image_paths, val_captions, config.val_data)
    save_data(test_image_paths, test_captions, config.test_data)


def write_json(data,data_path):
	with open(data_path,'w') as f:
		json.dump(data,f,indent=4)


def read_json(data_path):
	with open(data_path,'r') as f:
		data = json.load(f)
	return data


spacy_eng = spacy.load("en_core_web_sm")


class preprocess_caption:
	def __init__(self,captions,freq_threshold):
		self.captions = captions
		self.freq_threshold = freq_threshold


	def tokenizer_eng(self,text):
		return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]


	def get_vocab(self):
		vocab = []
		word2fre ={}
		self.max_length = 0
		for caption in self.captions:
			for idx,word in enumerate(self.tokenizer_eng(caption)):
				if word not in word2fre:
					word2fre[word] = 1
				else:
					word2fre[word]+=1
			if idx + 1 > self.max_length:
				self.max_length = idx + 1

		for word,freg in word2fre.items():
			if freg >= self.freq_threshold:
				vocab.append(word)
		self.vocab_size = len(vocab) + 1 # padding
		self.vocab = vocab
		return vocab


if __name__ == '__main__':
	# img_paths = glob.glob(os.path.join(config.img_folder,'*.jpg'))
	# print(len(img_paths))  # 8091 images

	# make_data()
	# split_data()
	# data = read_json(config.train_data)
	# captions = list(data.values())
	# captions = [x for y in captions for x in y]
	# vocab = preprocess_caption(captions,5)
	# print(len(vocab.get_vocab()))

	df = pd.read_csv(config.caption_path)
	rows = df.iloc[:,0:2]
	for row in rows:
		print(row)
