# coding=utf-8
import numpy as np
import struct
import os
import time

from model.base.layer import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer

class MLP(object):
    def __init__(self, batch_size=30, input_size=784, hidden1=256, hidden2=128, hidden3=64, out_classes=10, lr=0.01, max_epoch=30, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def shuffle_data(self):
        np.random.shuffle(self.train_data)

    def build_model(self):  # 建立网络结构
        # TODO：建立三层神经网络结构
        print('Building multi-layer perception model...')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.hidden3)
        self.relu3 = ReLULayer()
        self.fc4 = FullyConnectedLayer(self.hidden3, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3, self.fc4]

    def init_model(self):
        print('Initializing parameters of each layer in MLP...')
        for layer in self.update_layer_list:
            layer.init_param()

    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params = np.load(param_dir).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])
        self.fc4.load_param(params['w4'], params['b4'])

    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        params['w4'], params['b4'] = self.fc4.save_param()
        np.save(param_dir, params)

    def forward(self, input):  # 神经网络的前向传播
        # TODO：神经网络的前向传播
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        h3 = self.relu3.forward(h3)
        h4 = self.fc4.forward(h3)
        prob = self.softmax.forward(h4)
        return prob

    def backward(self):  # 神经网络的反向传播
        # TODO：神经网络的反向传播
        dloss = self.softmax.backward()
        dh4 = self.fc4.backward(dloss)
        dh3 = self.relu3.backward(dh4)
        dh3 = self.fc3.backward(dh3)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def train(self, features, labels):
        self.train_data = np.concatenate((features, labels), axis=1)
        max_batch = int(self.train_data.shape[0] / self.batch_size)
        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            acc_num = 0
            total_loss = 0
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                total_loss += loss
                self.backward()
                self.update(self.lr)

                # 统计训练集准确率
                pred_labels = np.argmax(prob, axis=1)
                acc_num += np.sum(pred_labels == batch_labels)

            print(f"Epoch {idx_epoch+1}: Loss = {total_loss/max_batch}, Accuracy = {acc_num/(max_batch*self.batch_size)}")

    def evaluate(self, feature):
        self.test_data = feature
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(self.test_data.shape[0]):
            image = self.test_data[idx, :]
            prob = self.forward(image)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx] = pred_labels
        return pred_results