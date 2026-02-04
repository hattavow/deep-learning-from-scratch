import os
import sys

sys.path.append(os.pardir)
import numpy as np

from common.functions import cross_entropy_error, softmax
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 重みの初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)

        loss = cross_entropy_error(y, t)
        return loss


net = simpleNet()
print("初期の重み:", net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)

print("予測結果:", p)

print(np.argmax(p))

t = np.array([0, 0, 1])  # 正解ラベル
loss = net.loss(x, t)
print("損失関数の値:", loss)

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)
