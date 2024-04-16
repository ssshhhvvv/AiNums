import time
from data import get_mnist
from funcs import *


class Network:

    def __init__(self):
        self.images, self.labels = get_mnist()
        self.w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
        self.w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
        self.b_h = np.zeros((20, 1))
        self.b_o = np.zeros((10, 1))
        self.l_r = 0.01
        self.epochs = 5
        self.nr_correct = 0

    def show_img(self, n, title):
        plt.title(title)
        plt.imshow(self.images[n].reshape(28, 28), cmap='Greys')

    def feed_forward(self, i):
        i = i.reshape((784, 1))
        h = sigmoid(np.dot(self.w_i_h, i) + self.b_h)
        o = sigmoid(np.dot(self.w_h_o, h) + self.b_o)
        return h, o

    def SGD(self):
        for epoch in range(self.epochs):
            for img, l in zip(self.images, self.labels):
                img.shape += (1,)
                l.shape += (1,)
                h, o = self.feed_forward(img)
                self.nr_correct += int(np.argmax(o) == np.argmax(l))
                delta_o = o - l
                self.w_h_o += -self.l_r * delta_o @ np.transpose(h)
                self.b_o += -self.l_r * delta_o
                delta_h = np.transpose(self.w_h_o) @ delta_o * (h * (1 - h))
                self.w_i_h += -self.l_r * delta_h @ np.transpose(img)
                self.b_h += -self.l_r * delta_h
            print(f"Acc: {round((self.nr_correct / self.images.shape[0]) * 100, 2)}%")
            self.nr_correct = 0
        while True:
            print('Введите число в диапозоне 0 - 59999')
            n = int(input())
            h, o = self.feed_forward(self.images[n])
            res = np.argmax(o)
            self.show_img(n, res)
            plt.show()
            time.sleep(1)


n = Network()
n.SGD()