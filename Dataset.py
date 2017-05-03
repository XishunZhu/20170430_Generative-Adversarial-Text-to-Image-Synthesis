#coding:utf-8
import os
import torch.utils.data as data
import torch
import random
from PIL import Image
from torch.utils.serialization import load_lua

class Dataset_cub(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.N = 5894
        self.target_images = ['' for i in range(self.N)]
        self.input_texts = torch.FloatTensor(self.N * 10, 1024)

        # now load the required data
        if self.train:
            train_cls = open(self.root + "/cub_icml/trainclasses.txt","r")
            train_class = train_cls.readlines()
            train_cls.close()
            k = 0
            for i in range(len(train_class)):
                a = self.root + "/cub_icml/" + train_class[i]
                b = str.strip(a, "\n")
                i_class = os.listdir(b)
                for j in range(len(i_class)):
                    j_class = i_class[j]
                    t7file = load_lua(b + '/' + j_class)
                    self.target_images[k] = self.root + '/CUB_200_2011/CUB_200_2011/images/' + t7file.img
                    for p in range(10):
                        self.input_texts[k*10 + p] = (t7file.txt)[p]
                    k = k + 1
        else:
            pass
        print(k)

    def __getitem__(self, index):

        input_txt = (self.input_texts[index*10 + random.randint(0, 9)] + self.input_texts[index*10 + random.randint(0, 9)] + self.input_texts[index*10 + random.randint(0, 9)] + self.input_texts[index*10 + random.randint(0, 9)]) / 4.0 # [torch.FloatTensor of size 1024]

        target_img = Image.open(self.target_images[index])
        wrong_num = random.randint(0, self.N - 1)
        wrong_img = Image.open(self.target_images[wrong_num])

        if self.transform is not None:
            target_img = self.transform(target_img)
            wrong_img = self.transform(wrong_img)
        if (target_img.size())[0] == 1:  # dealing with grayscale iamge
            target_img = torch.cat((target_img, target_img, target_img), 0)
        if (wrong_img.size())[0] == 1:  # dealing with grayscale iamge
            wrong_img = torch.cat((wrong_img, wrong_img, wrong_img), 0)
        return input_txt, target_img, wrong_img

    def __len__(self):
        return self.N


class Dataset_flower(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.N = 5878
        self.target_images = ['' for i in range(self.N)]
        self.input_texts = torch.FloatTensor(self.N * 10, 1024)
        self.raw_text = torch.FloatTensor(self.N * 10, 201)

        # now load the required data
        if self.train:
            train_cls = open(self.root + "/flowers_icml/trainclasses.txt","r")
            train_class = train_cls.readlines()
            train_cls.close()
            k = 0
            for i in range(len(train_class)):
                a = self.root + "/flowers_icml/" + train_class[i]
                b = str.strip(a, "\n")
                i_class = os.listdir(b)
                for j in range(len(i_class)):
                    j_class = i_class[j]
                    t7file = load_lua(b + '/' + j_class)
                    self.target_images[k] = self.root + '/102flowers/' + t7file.img
                    for p in range(10):
                        self.input_texts[k*10 + p] = (t7file.txt)[p]
                        self.raw_text[k*10 + p] = (t7file.char)[:,p]
                    k = k + 1
        else:
            pass

        print(k)

    def __getitem__(self, index):
        # input_txt = (self.input_texts[index*10 + random.randint(0, 9)] + self.input_texts[index*10 + random.randint(0, 9)] + self.input_texts[index*10 + random.randint(0, 9)] + self.input_texts[index*10 + random.randint(0, 9)]) / 4.0 # [torch.FloatTensor of size 1024]
        a = torch.randperm(10)
        input_txt = (self.input_texts[index*10 + a[0]] + self.input_texts[index*10 + a[1]] + self.input_texts[index*10 + a[2]] + self.input_texts[index*10 + a[3]]) / 4.0 # [torch.FloatTensor of size 1024]
        raw_txt = self.raw_text[index*10 + a[0]]
        target_img = Image.open(self.target_images[index])
        wrong_num = random.randint(0, self.N - 1)
        wrong_img = Image.open(self.target_images[wrong_num])

        if self.transform is not None:
            target_img = self.transform(target_img)
            wrong_img = self.transform(wrong_img)
        if (target_img.size())[0] == 1:  # dealing with grayscale iamge
            target_img = torch.cat((target_img, target_img, target_img), 0)
        if (wrong_img.size())[0] == 1:  # dealing with grayscale iamge
            wrong_img = torch.cat((wrong_img, wrong_img, wrong_img), 0)
        return input_txt, target_img, wrong_img, raw_txt

    def __len__(self):
        return self.N

class myDataset(data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, ext='png'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.ext = ext
        self.N = 0

        # now load the required data
        if self.train:
            files = os.listdir(self.root + "/train/input_imgs/")
            self.N = len(files)

            self.g_input_images = ['' for i in range(self.N)]
            self.g_input_texts = ['' for i in range(self.N)]
            self.target_images = ['' for i in range(self.N)]

            for i in range(len(files)):
                self.g_input_images[i] = self.root + 'train/input_imgs/' + files[i]
                self.target_images[i] = self.root + 'train/target_imgs/' + files[i]
                self.g_input_texts[i] = self.root + 'train/input_txts/' + files[i][0:-1*len(self.ext)] + 't7'
        else:
            pass

    def __getitem__(self, index):

        input_img = Image.open(self.g_input_images[index])
        input_txt = load_lua(self.g_input_texts[index])
        target_img = Image.open(self.target_images[index])

        if self.transform is not None:
            input_img = self.transform(input_img)

        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        input_txt = torch.mean(input_txt, 0)

        return input_img, input_txt, target_img

    def __len__(self):
        return self.N