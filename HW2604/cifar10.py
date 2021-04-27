!git clone https://github.com/YoongiKim/CIFAR-10-images

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, optim
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler


    

                                     
## Step 4
# Write your custom data loader. Define train, validation and test dataloader


# creating dataset module
class CIFAR10(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.X = self.df.image_path.values
        self.y = self.df.label.values
        self.transform = transform

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, idx):
        image = Image.open(self.X[idx])
        image = self.transform(image)
        label = self.y[idx]

        return image, label
        



def imshow(img):
  img = img/ 2 + 0.5
  plt.imshow(np.transpose(img, (1,2,0)))



'''def imshow(img):
  img = img/ 2 + 0.5
  plt.imshow(np.transpose(img, (1,2,0)))'''


    
    
    
## Step 6
# model = CNN(n_hidden_layers, n_output)

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3,32,3, padding=1)
    self.conv2 = nn.Conv2d(32,64,3, padding=1)
    self.conv3 = nn.Conv2d(64,128,3, padding=1)
    self.conv4 = nn.Conv2d(128,128,3, padding=1)
    self.conv5 = nn.Conv2d(128,256,3, padding=1)
    self.conv6 = nn.Conv2d(256,256,3, padding=1)
    self.pool =  nn.MaxPool2d(2,2)
    self.relu = nn.ReLU(inplace=True)
    self.fc1 = nn.Linear(4096,1024)
    self.fc2 = nn.Linear(1024,512)
    self.fc3 = nn.Linear(512,10)
    self.dropout1 = nn.Dropout(0.05)
    self.dropout2 = nn.Dropout(0.1)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)
    
    x = self.conv3(x)
    x = self.relu(x)
    x = self.conv4(x)
    x = self.relu(x)
    x = self.pool(x)
    x = self.dropout1(x)    

    x = self.conv5(x)
    x = self.relu(x)
    x = self.conv6(x)
    x = self.relu(x)
    x = self.pool(x)
    
    x = x.view(-1, 4096)
    x = self.dropout2(x)
    x = self.relu(self.fc1(x))
    x = self.dropout2(x)
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x



## Step 7
# Define loss and solver
# criterion = ...
# optimizer = ...
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)



## Step 8
# train_with_validation
# train(n_epoch, model_filename, criterion, optimizer)

def train(epoch, model_name, criterion, optimizer, trainldr, validldr):
  valid_loss_min = np.Inf

  for e in range(epoch):
    train_loss = 0
    valid_loss = 0

    model.train()
    for images, labels in trainldr:
      images, labels = images.cuda(), labels.cuda()
      log_probs = model(images)
      loss = criterion(log_probs, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss += loss.item()*len(images)
    else:
      model.eval()
      with torch.no_grad():
        for images, labels in validldr:
          images, labels = images.cuda(), labels.cuda()
          log_probs = model(images)
          loss = criterion(log_probs, labels)
          valid_loss += loss.item()*len(images)

    train_loss = train_loss/len(trainldr.sampler)
    valid_loss = valid_loss/len(validldr.sampler)

    print("epoch: {}/{}".format(e+1, epoch),
          "train_loss: {:.4f}".format(train_loss),
          "valid_loss: {:.4f}".format(valid_loss))
    
    if valid_loss <= valid_loss_min:
      path = F"/content/{model_name}"
      torch.save(model.state_dict(), path)
      valid_loss_min = valid_loss
      


## Step 9
# Evaluation with inference: load model
# performance = test(model_filename) # total accuracy

def test(model_name):
    path = F"/content/{model_name}"
    model.load_state_dict(torch.load(path))

    test_loss = 0
    test_accuracy = 0
    
    model.eval()
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item()
        probs = torch.exp(output)
        top_prob, top_class = probs.topk(1, dim=1)
        equals = top_class == labels.view(top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    test_loss = test_loss/len(test_loader)
    test_accuracy = test_accuracy/len(test_loader)
    print('Test Loss: {:.3f}\n'.format(test_loss),
          'Test Accuracy: {:.3f}'.format(test_accuracy))
    
    fp = open(performance_path+"/performance.txt", "x")
    fp.write('Test Loss: {:.3f}... Test Accuracy: {:.3f}'.format(test_loss, test_accuracy))
    fp.close()
    
          

if __name__== "__main__":

## Step 2
# Activate CUDA on Google Colab

	train_on_gpu = torch.cuda.is_available()

	if not train_on_gpu:
    		print('CUDA is not available.  Training on CPU ...')
	else:
    		print('CUDA is available!  Training on GPU ...')
	
## Step 3
# Define data transformations to facilitate data augmentation and normalization
# E.g., flip, rotation, translation, to_tensor, normalize

	train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                          (0.2023, 0.1994, 0.2010))
                                      ])

	test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2023, 0.1994, 0.2010)),
                                     ])
                                     
	
	test_path = '/content/CIFAR-10-images/test'
	test_paths = list(paths.list_images(test_path))
	test_data = pd.DataFrame(columns=['image_path', 'label'])
	test_labels = []
	#classes = []
	for i, image_path in enumerate(test_paths):
	    test_data.loc[i, 'image_path'] = image_path
	    test_label = image_path[len(test_path):].split('/')[1]
	    test_labels.append(test_label)
	    #classes.append((test_label))
	
	test_labels = np.array(test_labels)
	# one-hot encoding
	labels = LabelBinarizer().fit_transform(test_labels)
	
	for i in range(len(labels)):
	    idx = np.argmax(labels[i])
    	test_data.loc[i,"label"] = idx
	
	test_data = test_data.sample(frac=1).reset_index(drop=True) #shuffle the dataset
	test_data.to_csv(test_path+'data.csv', index=False)
	
	
	train_path = '/content/CIFAR-10-images/train'
	train_paths = list(paths.list_images(train_path))
	train_data = pd.DataFrame(columns=['image_path', 'label'])
	train_labels = []
	
	for i, image_path in enumerate(train_paths):
	    train_data.loc[i, 'image_path'] = image_path
	    train_label = image_path[len(test_path):].split('/')[1]
	    train_labels.append(train_label)
	
	train_labels = np.array(train_labels)
	# one-hot encoding
	labels = LabelBinarizer().fit_transform(train_labels)
	
	for i in range(len(labels)):
	    idx = np.argmax(labels[i])
	    train_data.loc[i,"label"] = idx
	    
	train_data = train_data.sample(frac=1).reset_index(drop=True) #shuffle the dataset
	train_data.to_csv(train_path+'data.csv', index=False)
	
	valid = 0.2
	batch_size = 20
	
	train_data = CIFAR10(train_path+'data.csv', transform = train_transform)
	test_data = CIFAR10(test_path+'data.csv', transform = train_transform)
	
	num_train = len(train_data)
	indices = list(range(num_train))
	split = int(valid * num_train)
	train_idx, valid_idx = indices[split:], indices[:split]
	
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)
	
	train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
	valid_loader = DataLoader(train_data, sampler=valid_sampler, batch_size=batch_size)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle = True)
	
	classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



	
## Step 5 (OPTIONAL)
# Visualize a batch of train (2 x n_class) and test data (2 x n_classes)

	images, labels = next(iter(train_loader))
	images = images.numpy()

	fig = plt.figure(figsize=(25,4))
	
	
	for idx in np.arange(batch_size):
    	ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    	imshow(images[idx])
    	ax.set_title(classes[labels[idx]])
    	#ax.title.set_text('Train Data')


	images, labels = next(iter(test_loader))
	images = images.numpy()

	fig1 = plt.figure(figsize=(25,4))
	
	for idx in np.arange(batch_size):
    	ax1 = fig1.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    	#ax1.title.set_text('Test data')
    	imshow(images[idx])
    	ax1.set_title(classes[labels[idx]])
	
	
	model_name = "cifar10_cnn"
	train(epoch, model_name, criterion, optimizer, train_loader, valid_loader)
	
	performance_path = input("enter the to save the performance file:")
	test('cifar10_cnn')    










































                                         
