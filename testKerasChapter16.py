#!/usr/bin/env python
# coding: utf-8



# In[5]:


import PIL 
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.imagenet_utils import decode_predictions 
import matplotlib.pyplot as plt 
import numpy as np 
from keras.applications.resnet50 import ResNet50 
from keras.applications import resnet50 


# In[6]:


filename = 'banana.jpg'
## load an image in PIL format 
original = load_img(filename, target_size=(224, 224))
print('PIL image size',original.size)


# In[7]:


plt.imshow(original)


# In[8]:


#convert the PIL image to a numpy array 
numpy_image = img_to_array(original) 
plt.imshow(np.uint8(numpy_image))


# In[9]:


print('numpy array size',numpy_image.shape)


# In[10]:


# Convert the image / images into batch format 
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)


# In[11]:


# prepare the image for the resnet50 model 
processed_image = resnet50.preprocess_input(image_batch.copy()) 

# create resnet model 
resnet_model = resnet50.ResNet50(weights='imagenet')

# get the predicted probabilities for each class 
predictions = resnet_model.predict(processed_image)

# convert the probabilities to class labels 
label = decode_predictions(predictions)

print(label)


# In[ ]:

# [[('n01728920', 'ringneck_snake', 0.28570452),
 # ('n01734418', 'king_snake', 0.18434986),
 # ('n03134739', 'croquet_ball', 0.18004729),
 # ('n01740131', 'night_snake', 0.11061155),
 # ('n01748264', 'Indian_cobra', 0.05979025)]]


