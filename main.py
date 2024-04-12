#!/usr/bin/env python
# coding: utf-8

# In[94]:


import numpy as np
import matplotlib.pyplot as plt
import time
from data import get_mnist

images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_h = np.zeros((20, 1))
b_o = np.zeros((10, 1))

l_r = 0.01
epochs = 5
nr_correct = 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# In[97]:


def show_img(n, title):
    plt.title(title)
    plt.imshow(images[n].reshape(28, 28), cmap='Greys')


# In[ ]:


def feed_forward(i):
    i = i.reshape((784, 1))
    h = sigmoid(np.dot(w_i_h, i) + b_h)
    o = sigmoid(np.dot(w_h_o, h) + b_o)
    return h, o


# In[103]:


for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        h, o = feed_forward(img)
        nr_correct += int(np.argmax(o) == np.argmax(l))
        delta_o = o - l
        w_h_o += -l_r * delta_o @ np.transpose(h)
        b_o += -l_r * delta_o
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -l_r * delta_h @ np.transpose(img)
        b_h += -l_r * delta_h
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0
while True:
    print('Введите число в диапозоне 0 - 59999')
    n = int(input())
    h, o = feed_forward(images[n])
    res = np.argmax(o)
    show_img(n, res)
    plt.show()
    time.sleep(1)

