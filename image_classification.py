#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.executable


# In[2]:


pip install tensorflow keras matplotlib scikit-learn


# In[3]:


pip install --upgrade pip


# In[2]:


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# In[3]:


get_ipython().system('pip show tensorflow keras matplotlib scikit-learn')


# In[4]:


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# In[5]:


train_images, test_images = train_images / 255.0, test_images / 255.0


# In[6]:


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']


# In[7]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# In[8]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[9]:


model.summary()


# In[10]:


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


# In[11]:


model.summary()


# In[12]:


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# In[13]:


history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


# In[14]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# In[15]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


# In[16]:


print(test_acc)


# In[20]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image


img_path = "animal2.jpg"  
img = Image.open(img_path)

img = img.resize((32, 32))

img_array = np.array(img) / 255.0  
img_array = np.expand_dims(img_array, axis=0)  

plt.imshow(img)
plt.axis("off")
plt.show()


# In[21]:


predictions = model.predict(img_array)

predictions = tf.nn.softmax(predictions)

predicted_class = np.argmax(predictions)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Predicted Class: {class_names[predicted_class]}")


# In[ ]:




