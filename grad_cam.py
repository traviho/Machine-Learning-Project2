# -*- coding: utf-8 -*-
"""grad_cam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BhuCthYX8oFPZVNLb4IOBkzQLV9Lk9fk
"""

from torchvision import models
import torchvision
import numpy as np
import cv2
from torch.autograd import Variable
import torch

gradients = []


def collect_gradient(gradient):
  #print(gradient)
  gradients.append(gradient)
  
def extract_features(model, layers, x):
  print(model._modules.items())
  outputs = []
  for n, m in model._modules.items():
    x = m(x)
    
    if n in layers:
      
      x.register_hook(collect_gradient)
      outputs.append(x)
  return x, outputs

def get_layer_activations(model, layers, x):
  output, activations = extract_features(model.features, layers, x)
  output = model.classifier(output.view(output.size(0), -1))
  
  return activations, output

def img_to_tensor(img):

  img = cv2.resize(img, (224,224))
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
  img = np.ascontiguousarray(np.transpose(img, (2,0,1)))
  tensor = torch.from_numpy(img)
  tensor.unsqueeze_(0)
  return Variable(tensor, requires_grad=True)

def overlay_heatmap(filename, mask, input_image):
  
  heatmap = cv2.resize(np.uint8(mask * 255), (224, 224))
  heatmap = cv2.applyColorMap(normalize_heatmap(heatmap), cv2.COLORMAP_JET)
  img = cv2.resize(input_image, (224,224)) 
  
  
  overlayed = cv2.addWeighted(heatmap, 0.3, img, 0.7, 0)
  
  cv2.imwrite(filename, overlayed)
 
  
def normalize_heatmap(heatmap):
  
  heatmap = np.float32(heatmap) 
  heatmap = (heatmap / np.max(heatmap))
  heatmap = (heatmap - np.min(heatmap)) * 255
  
  return np.uint8(heatmap)
  

input_image = cv2.imread("2008_000008.jpg", 1)

net = models.vgg19(pretrained=True)

net.eval()

input = img_to_tensor(np.float32(input_image) /255)

activs, output = get_layer_activations(net, ["36"], input)

###choses the index for the output with the highest activation
i = np.argmax(output.data.numpy())

output_mask = np.zeros((1, output.size()[-1]), dtype=np.float32)
output_mask[0][i] = np.float32(1)
output_mask = Variable(torch.from_numpy(output_mask), requires_grad=True)
output_mask = torch.sum(output * output_mask)

net.features.zero_grad()
net.classifier.zero_grad()

output_mask.backward(retain_graph=True)

gradients_last_layer = np.array(gradients[-1])

last_layer_activ = activs[-1].data.numpy()[0,:]

weights = np.mean(gradients_last_layer, axis = (2, 3))[0, :]

cam_mask = np.zeros(last_layer_activ.shape[1 : ], dtype=np.float32)

for index, weight in enumerate(weights):
  result = weight * last_layer_activ[index,:,:]
  cam_mask += result
    
overlay_heatmap("grad_cam_vgg19.jpg", cam_mask, input_image)