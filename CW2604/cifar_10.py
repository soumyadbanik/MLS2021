from imutils import paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

! git clone https://github.com/YoongiKim/CIFAR-10-images

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

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    

# getting all the image paths
train_path = '/content/CIFAR-10-images/train'
tr_paths = list(paths.list_images(train_path))

test_path = '/content/CIFAR-10-images/test'
test_paths = list(paths.list_images(test_path))

test_data = pd.DataFrame(columns=['image_path', 'target'])
test_labels = []

train_data = pd.DataFrame(columns=['image_path', 'target'])
train_labels = []

for i, image_path in enumerate(test_image_paths):
    test_data.loc[i, 'image_path'] = image_path
    test_label = image_path[len(test_path):].split('/')[1]
    test_labels.append(test_label)

test_data['target'] = test_labels

for i, image_path in enumerate(train_image_paths):
    train_data.loc[i, 'image_path'] = image_path
    train_label = image_path[len(test_path):].split('/')[1]
    train_labels.append(train_label)

train_data['target'] = train_labels

# creating a csv file from the dataframe
train_data = train_data.sample(frac=1).reset_index(drop=True) #shuffle the dataset
train_data.to_csv(train_path+'/train.csv', index=False)

test_data = test_data.sample(frac=1).reset_index(drop=True) #shuffle the dataset
test_data.to_csv(test_path+'/test.csv', index=False)

df = pd.read_csv(train_path+'/train.csv')
X = df.image_path.values
y = df.target.values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)


## Step 4
# Write your custom data loader. Define train, validation and test dataloader

## Step 5 (OPTIONAL)
# Visualize a batch of train (2 x n_class) and test data (2 x n_classes)

## Step 6
# model = CNN(n_hidden_layers, n_output)

## Step 7
# Define loss and solver
# criterion = ...
# optimizer = ...

## Step 8
# train_with_validation
# train(n_epoch, model_filename, criterion, optimizer)

## Step 9
# Evaluation with inference: load model
# performance = test(model_filename) # total accuracy

## Step 10
# Push the .py files to MLS2021 github with branchname CW2604
# There should be a main.py
# performance.txt



