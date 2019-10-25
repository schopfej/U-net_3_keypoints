
# coding: utf-8
# In[131]:
import scipy.ndimage.filters as fi
import numpy as np
from torch.autograd import Variable
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from workspace_utils import active_session
import torch
import torch.nn as nn
import torch.nn.functional as F
## TODO: Once you've define the network, you can instantiate it
# one example conv layer has been provided for you
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
from scratch import UNet
# model = UNetWithResnet50Encoder()
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import glob
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader



le = preprocessing.LabelEncoder()
net = UNet(3, in_channels=3, depth=5, merge_mode='concat')

def gkern2(tensor_coord, size_x=848, size_y=464, sig_x=3, sig_y=3):  # by default like size of a iPhone video
    """Returns a 2D Gaussian kernel array."""
    # tensor_coord = tensor_coord[0]
    # create nxn zeros
    inp = np.zeros((size_y, size_x))
    # tensor_coord = tensor_coord[~np.isnan(tensor_coord).any(axis=1)]

    tensor_coord = tensor_coord[(tensor_coord < size_x).any(1)]
    tensor_coord = tensor_coord[(0 < tensor_coord).any(1)]
    # size x and y are equal..... when there is a Nan value, it is arbitrarily high value
    # set element at the middle to one, a dirac delta

    print("did the croping successful habennadlfkjadfklaj", tensor_coord)
    gaussian_3d = np.zeros((3, size_y, size_x))
    inp[tensor_coord[:, 1], tensor_coord[:, 0]] = 1
    sigma = [sig_y, sig_x]
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    gaussian = fi.gaussian_filter(inp, sigma)
    #gaussian[gaussian > 0] = 1
    for i, row in enumerate(tensor_coord):
        gaussian_3d[i, tensor_coord[i], tensor_coord[i, 0]] = 1
        gaussian_3d[i] = fi.gaussian_filter(gaussian_3d[i], sigma)


    inp2 = np.transpose(gaussian_3d)
    inp3 = np.swapaxes(inp2, 0, 1).reshape(-1, 3)

    #b = np.zeros_like(inp3)
    #b[np.arange(len(inp3)), inp3.argmax(1)] = 1
    #b = b*(1,2,3,4)
    #b = b[b!=0]
    #class_labels = le.inverse_transform(b)
    #class_labels = np.argmax(b,axis=1).reshape(-1,4)


    #plt.imshow(gaussian)
    #plt.show()
    return gaussian, inp3


# In[133]:
# b = [not bool for bool in a]
# gaussian-smooth the dirac, resulting in a gaussian filter mask
# ## CNN Architecture
#
# Recall that CNN's are defined by a few types of layers:
# * Convolutional layers
# * Maxpooling layers
# * Fully-connected layers
#
#
# In[134]:
# import the usual resources


## TODO: define the data_transform using transforms.Compose([all tx's, . , .])
# order matters! i.e. rescaling should come before a smaller crop
# testing that you've defined a transform
data_transform = transforms.Compose([Rescale(224),
                                     # RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
assert (data_transform is not None), 'Define a data_transform'
# In[ ]:
# In[136]:


# In[137]:
###########load the csv
# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(224),
                                     # RandomCrop(222),
                                     Normalize(),
                                     ToTensor()])
# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file=('/home/jonathan/Project_Eye/data/try_harder_cleaned.csv'),
                                             root_dir='/home/jonathan/Project_Eye/data/training/',
                                             transform=data_transform)
print('Number of images: ', len(transformed_dataset))
# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())

# ## Batching and loading data
#
# In[138]:
# load training data in batches
batch_size = 1
train_loader = DataLoader(transformed_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
# ## Before training
#
# Take a look at how this model performs before it trains. You should see that the keypoints it predicts start off in one spot and don't match the keypoints on a face at all! It's interesting to visualize this behavior so that you can compare it to the model after training and see how the model has improved.
# #### Load in the test dataset
# The test dataset is one that this model has *not* seen before, meaning it has not trained with these images. We'll load in this test data and before and after training, see how your model performs on this set!
# To visualize this test data, we have to go through some un-transformation steps to turn our images into python images from tensors and to turn our keypoints back into a recognizable range.
# In[139]:
# load in the test data, using the dataset class
# AND apply the data_transform you defined above
# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='/home/jonathan/Project_Eye/data/try_harder_cleaned.csv',
                                      root_dir='/home/jonathan/Project_Eye/data/training/',
                                      transform=data_transform)
## create the transformed dataset
# transformed_dataset = FacialKeypointsDataset(csv_file=('/home/jonathan/Project_Eye/data/try_harder_cleaned.csv'),
# root_dir='/home/jonathan/Project_Eye/data/training/',
#
# In[140]:
# load test data in batches
batch_size = 1
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)
# ## Apply the model on a test sample
# Problem is, that we have now the image as torch type and not as numpy array anymore

def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='x', c='red')


data_transform = transforms.Compose([Rescale(224),
                                     # RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='/home/jonathan/Project_Eye/data/try_harder_cleaned.csv',
                                             root_dir='/home/jonathan/Project_Eye/data/training/',
                                             transform=data_transform)




def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


# #### Un-transformation
#
# Next, you'll see a helper function. `visualize_output` that takes in a batch of images, predicted keypoints, and ground truth keypoints and displays a set of those images and their true/predicted keypoints.
#
# This function's main role is to take batches of image and keypoint data (the input and output of your CNN), and transform them into numpy images and un-normalized keypoints (x, y) for normal display. The un-transformation process turns keypoints and images into numpy arrays from Tensors *and* it undoes the keypoint normalization done in the Normalize() transform; it's assumed that you applied these transformations when you loaded your test data.
# In[71]:
# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    for i in range(batch_size):
        plt.figure(figsize=(20, 10))
        ax = plt.subplot(1, batch_size, i + 1)
        # un-transform the image data
        image = test_images[i].data  # get the image from it's Variable wrapper
        image = image.numpy()  # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))  # transpose to go from torch to numpy image
        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts * 50.0 + 100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

        plt.axis('off')
    plt.show()





## TODO: Define the loss and optimization
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adam(params=net.parameters(), lr=7.1)
#
#
#
#
# ## Training and Initial Observation
# Now, you'll train on your batched training data from `train_loader` for a number of epochs.
# To quickly observe how your model is training and decide on whether or not you should modify it's structure or hyperparameters, you're encouraged to start off with just one or two epochs at first. As you train, note how your the model's loss behaves over time: does it decrease quickly at first and then slow down? Does it take a while to decrease in the first place? What happens if you change the batch size of your training data or modify your loss function? etc.
#
# Use these initial observations to make changes to your model and decide on the best architecture before you train for many epochs and create a final model.
# ### find out the size of the prepared image
# later we use it to do a gaussian probability fct.
# In[169]:
for i, sample_size in enumerate(transformed_dataset):
    if i == 0:
        images = sample['image'].squeeze_()
        size_x, size_y = images.shape[:2]
        print(size_x)
        resized_coord = sample['keypoints'].squeeze_()
        break
# ### test wheter we can do a gaussian probability image
#
# 1. find out which coordinate
# 2. label this point and then do a gaussian convolution
# In[170]:
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

for i, sample in enumerate(transformed_dataset):
    # get sample data: images and ground truth keypoints
    images = sample['image']
    # key_pts = sample['keypoints'].float()
    key_pts = sample['keypoints']
    # convert images to FloatTensors
    # images = images.type(torch.FloatTensor)
    # print("this sample has the size:", sample['keypoints'].shape)
    print("now a little bit further:", sample['keypoints'])
    # forward pass to get net output
    if i > 0:
        break
    ####################################################################################
    ####################################################################################
    ##### turn our 3 coordinates to a gaussian probability image and then compare it with the predicted unet image
    keypoints_3 = np.array(sample['keypoints'])
    prob, prob_3d = gkern2(tensor_coord=keypoints_3.astype("int"), size_x=size_y, size_y=size_y, sig_x=10, sig_y=10)
    # plt.plot(prob)
    # plt.show()
    nx, ny = prob.shape[1], prob.shape[0]
    x = range(nx)
    y = range(ny)
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    ha.view_init(30, 100)
    X, Y = np.meshgrid(x, y)
    ha.plot_surface(X, Y, prob, cmap=cm.summer)
    plt.title('Normalized gauss filter of size 21x21')
    plt.show()

# # Everything looks fine:
# ## -the image is reshaped
# ## -the Gaussian probability image is made out from keypoitns
# ## Now: let the image through the Unet ....................then compare these two probability images with a loss function
# In[171]:





#criterion = nn.BCEWithLogitsLoss
#criterion1 = nn.Sigmoid()
#criterion = nn.L1Loss
#optimizer = optim.Adam(params=net.parameters(), lr=0.0025)
softm = nn.Softmax(dim=1)
#criterion = nn.CrossEntropyLoss()
sigm = nn.Sigmoid()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
net = UNet(3, in_channels=3, depth=5, merge_mode='concat')



def train_net(n_epochs):
    # prepare the net for training
    net.train()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(transformed_dataset):
            # get the input images and their corresponding labels
            images = data['image']
            images = images.reshape(1, 3, size_y, size_y)
            # images = Variable(torch.FloatTensor(images.reshape(1,1,size_x,size_y)))
            key_pts = data['keypoints']
            if np.isnan(key_pts).any():
                break
            print("the images for prediction:", images.shape)
            print("the images type:", type(images))
            keypoints_3 = np.array(data['keypoints'])
            prob, prob_3d = gkern2(tensor_coord=keypoints_3.astype("int"), size_x=size_y, size_y=size_y, sig_x=30, sig_y=30)


            # x = Variable(torch.FloatTensor(1, 1, images))
            images = images.type(torch.FloatTensor)
            # print("images after torch",images)
            # forward pass to get outputs)
            optimizer.zero_grad()
            out = net(images)
            ################## Shape of output #########################################################
            ############################################################################################
            # permute is like np.transpose: (N, C, H, W) => (H, W, N, C)
            # contiguous is required because of this issue: https://github.com/pytorch/pytorch/issues/764
            # view: reshapes the output tensor so that we have (H * W * N, num_class)
            # NOTE: num_class == C (number of output channels)
            # output = output.permute(2, 3, 0, 1).contiguous().view(-1, num_classes)
            out = out.permute(2, 3, 0, 1).contiguous().view(-1, 3)
            ### ###########################
            ## Cross entropy (input,target/labels)
            ###################################
            ###### Input: (N,C) where C = number of classes
            ########## Target: (N) where each value is 0≤targets[i]≤C−1
            ########## Output: scalar
            #prob = preprocessing.minmax_scale(prob_3d,feature_range=(0, 1))
            prob = torch.from_numpy(prob_3d)
            prob = Variable(prob.long())
            prob = prob.squeeze_()
            out = out.squeeze_()
            # out = out.reshape(-1, 1)
            # out = out.permute(2, 3, 0, 1).contiguous().view(-1, 3)
            # torch.sum(out)
            #pred = sigmoid(out)
            #pred = softm(pred)
            #out.append(hidden[0])
            #pred = prob.reshape(-1,4)
            loss = criterion(out, prob_3d)
            #out = criterion1(out)
            #out = softm(out)
            ###########activation

            # loss.backward()
            #optimizer.zero_grad()  # otherwise the parameters would be cumulated...
            # backward pass to calculate the weight gradients
            loss.backward()  ###############################################calculate gradients
            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss))
                running_loss = 0.0
                params = list(net.parameters())
                print("the len of the param", len(params))
                print("the param size is", params[0].size())  # conv1's .weight
                print(params[0])
    print('Finished Training')


train_net(2)

