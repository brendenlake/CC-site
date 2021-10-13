from __future__ import print_function, division
import math
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import euclidean,cosine

# Pytorch code for extracting the top representations from a pre-trained convnet

# Compares all images with an example image (in subfolder 'example'), using cosine distance
# Note that this code does some scaling and cropping. Plot uses reconstructed images.

use_gpu = torch.cuda.is_available()

# Image normalization (all pre-trained models use the same normalization)
T = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# T = transforms.Compose([
#     transforms.Scale(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

class Identity(nn.Module):
    # Simple neural net module that returns input tensor
    #  useful for extracting tensor at particular layers
    def forward(self, inputs):
        return inputs

def imshow(inp, title=None):
    # Visualize training image after pre-processing
    # 
    #  Input
    #    inp : numpy 3D tensor, after pre-processing
    #    title : optional title for plot
    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def get_features(data_dir,model_type):
    # Input
    #   data_dir : directory which contains image sub-directories (one for each class technically...)
    #   model_type : which type of convnet ['resnet18','vgg11', 'resnet152']
    #
    # Output
    #   output : [nimg x nfeat tensor] returns extracted features, with each image as a row
    #   images : [nimg x 3 x 224 x 224 tensor] raw images

    # Process the input data
    image_dataset = datasets.ImageFolder(data_dir, T)
    class_names = image_dataset.classes
    dataset_size = len(image_dataset)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=dataset_size, shuffle=False, num_workers=1)

    # Get a batch of data, which should hold whole dataset
    inputs, labels = next(iter(dataloader))
    if use_gpu:
        inputs = inputs.cuda()
        labels = labels.cuda()

    # Load the model and replace layers with identity function to extract the right outputs
    # in each case, we are grabbing the top convolutional layer..
    if model_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        # if before_maxpool:
        #     model.avgpool = Identity()
        model.fc = Identity()
    elif model_type == 'resnet152':
        model = torchvision.models.resnet152(pretrained=True)
        # if before_maxpool:
        #     model.avgpool = Identity()
        model.fc = Identity()    
    elif model_type == 'vgg11':
        model = torchvision.models.vgg11(pretrained=True)
        model.classifier = Identity() # keep just the top conv layer
    else:
        assert False
    if use_gpu:
        model = model.cuda()

    # forward pass
    model.eval()
    outputs = model(inputs)
    return outputs, inputs

def similarity_gradient(data_dir,model_type,idx_sel=0,dist_type='euclidean'):
    # Input
    #   data_dir : directory which contains image sub-directories (one for each class technically...)
    #   model_type : which type of convnet ['resnet18','vgg11', 'resnet152']

    outputs,inputs = get_features(data_dir,model_type)
    outputs = outputs.data.numpy()
    inputs = inputs.cpu().data.numpy()

    # get the vector of responses
    v_example = outputs[idx_sel] # baseline for comparison
    v_choices = outputs # all other images
    img_example = inputs[idx_sel]
    img_choices = [img for idx,img in enumerate(inputs)]

    # get cosine distance for each
    if dist_type == 'cosine':
        mydist = cosine
    elif dist_type == 'euclidean':
        mydist = euclidean
    dist = [mydist(v,v_example) for v in v_choices]
    dist = [round(s,4) for s in dist]

    # show example
    nrow = math.ceil(math.sqrt(len(img_choices)))
    ax = plt.subplot(nrow, nrow, 1)
    ax.axis('off')
    imshow(img_example)
    plt.title('example')

    # show choices
    for j in range(len(img_choices)):
        ax = plt.subplot(nrow, nrow, j+2)
        ax.axis('off')
        imshow(img_choices[j])
        plt.title('dist='+str(dist[j]))
    plt.show()

if __name__ == "__main__":
    similarity_gradient('data','vgg11')