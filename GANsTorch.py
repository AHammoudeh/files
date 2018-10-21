# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:02:32 2018

@author: AH
"""
import numpy as np
import cv2
import os
import glob
from random import randint
import scipy.io as sio

import argparse
import keras
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch


#%%
img = np.zeros(shape=[280,320])
r = 150.0 / img.shape[1]
dim = (int(img.shape[0] * r),150)

class Paramtrs:
  n_epochs = 100 #help='number of epochs of training')
  batch_size = 5 #help='size of the batches')
  lr= 0.0002 #help='adam: learning rate')
  b1= 0.5 #help='adam: decay of first order momentum of gradient')
  b2= 0.999 #help='adam: decay of first order momentum of gradient')
  n_cpu= 8 #help='number of cpu threads to use during batch generation')
  latent_dim=100 #help='dimensionality of the latent space')
  img_size0=dim[1] #help='size of each image dimension')
  img_size1=dim[0] #help='size of each image dimension')
  channels=1 #help='number of image channels')
  sample_interval= 40 #help='interval between image sampling')
  
opt = Paramtrs()

#%%

img_dir = "./Iris_V1" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*bmp')
files = glob.glob(data_path)

owner = [np.random.randint(0, 107), np.random.randint(0, 107)]

while owner[0]==owner[1]:
  owner = [np.random.randint(0, 107), np.random.randint(0, 107)]


files_training = sorted(files)[7*owner[0]:7*(owner[0]+1)].copy()
files_training = files_training+sorted(files)[7*owner[1]:7*(owner[1]+1)]
files_testing =[]

for i in owner:
    testing_file_1 = sorted(files)[7*i][:-7]+'1_'+str(randint(1,3))+'.bmp'
    files_testing.append(testing_file_1)
    files_training.remove(testing_file_1)
    testing_file_2 = sorted(files)[7*i][:-7]+'2_'+str(randint(1,4))+'.bmp'
    files_testing.append(testing_file_2)
    files_training.remove(testing_file_2)
    
files_training = np.array(sorted(files_training))
files_testing = np.array(sorted(files_testing))

data_training = np.empty(shape=[0,opt.img_size0,opt.img_size1]) 
label_training = np.empty(shape=[0,1])

for f2 in files_training:
    img = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    data_training = np.concatenate((data_training, [img]))
    label_training = np.append(label_training,f2[-11:-8])

data_testing = np.empty(shape=[0,opt.img_size0,opt.img_size1]) 
label_testing = np.empty(shape=[0,1])

for f2 in files_testing:
    img = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    data_testing = np.concatenate((data_testing, [img]))
    label_testing =  np.append(label_testing,f2[-11:-8])  


def data_scale(data, Label):
    return (data/255,  Label.astype('int32')-1)

trainx, trainy = data_scale(data_training, label_training)
trainx2=trainx.reshape([trainx.shape[0],1, trainx.shape[1], trainx.shape[2]])

testx, testy = data_scale(data_testing, label_testing)
testx=testx.reshape([testx.shape[0],1, testx.shape[1], testx.shape[2]])

#%%
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

datagen.fit(trainx2)

x_batches = trainx2
y_batches = trainy

#%%
epochs = 10
for e in range(epochs):
	print('Epoch', e)
	batches = 0
	per_batch = 10
	for x_batch, y_batch in datagen.flow(trainx2, trainy, batch_size=per_batch):
		x_batches = np.concatenate((x_batches, x_batch), axis = 0)
		y_batches = np.concatenate((y_batches, y_batch), axis = 0)
		batches += 1
		if batches >= len(trainx2) / per_batch:
			# we need to break the loop by hand because
			# the generator loops indefinitely
			break
del x_batch,y_batch, i, e, epochs,f2,batches, img_dir, per_batch
#%%
a = np.zeros(y_batches.shape)
a[y_batches==owner[0]]=1
x_batches = torch.from_numpy(x_batches)
y_batches = torch.from_numpy(a)


class IrisDataset():
  def __init__(self,x,y):
    self.len = y.shape[0]
    self.x_data = x
    self.y_data = y
  def __getitem__(self, index):
    return self.x_data[index],self.y_data[index]
  def __len__(self):
    return self.len

dataset1 = IrisDataset(x_batches,y_batches)

dataloader = torch.utils.data.DataLoader(dataset1, batch_size=opt.batch_size, shuffle=True)


#%%
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
#%%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding= 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        #ds_size0 = opt.img_size0
        #ds_size1 = opt.img_size1
        self.adv_layer = nn.Sequential( nn.Linear(11520, 1),
                                        nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

    
#%%
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size0 = opt.img_size0
        self.init_size1 = opt.img_size1
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size0*self.init_size1))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size0, self.init_size1)
        img = self.conv_blocks(out)
        return img
    
#%%
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

#%%

for epoch in range(opt.n_epochs): 
    for i, (imgs, labels) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor), requires_grad=False)
        real_labels = Variable(labels.type(Tensor), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        output = discriminator(real_imgs)
        d_loss = adversarial_loss(output, real_labels)

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item()))

        batches_done = epoch * len(dataloader) + i
        #if batches_done % opt.sample_interval == 0:
            #save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

