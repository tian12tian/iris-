# coding:utf-8
import os
import codecs
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, GRU, BatchNormalization, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from bert.extract_feature import BertVector
from sklearn.metrics import f1_score, recall_score, precision_score


"""
将bert预训练模型（chinese_L-12_H-768_A-12）放到当前目录下
基于bert句向量的文本分类：基于Dense的微调
"""


class BertClassification(object):
    def __init__(self,
                 nb_classes=3,
                 dense_dim=256,
                 max_len=100,
                 batch_size=128,
                 epochs=50,
                 train_corpus_path="data/train.csv",
                 test_corpus_path="data/dev.csv",
                 weights_file_path="./model/weights_fc.h5"):
        self.nb_classes = nb_classes
        self.dense_dim = dense_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights_file_path = weights_file_path
        self.train_corpus_path = train_corpus_path
        self.test_corpus_path = test_corpus_path

        self.nb_samples = 17  # 样本数    D:\NLP项目\bert模型\chinese_L-12_H-768_A-12
        self.bert_model = BertVector(pooling_strategy="REDUCE_MEAN",
                                     max_seq_len=self.max_len,
                                     bert_model_path="chinese_L-12_H-768_A-12",
                                     graph_tmpfile="./tmp_graph_xxx")

    def text2bert(self, text):
        """ 将文本转换为bert向量  """
        vec = self.bert_model.encode([text])
        return vec["encodes"][0]

    def data_format(self, lines):
        """ 将数据转换为训练格式，输入为列表  """
        X, y = [], []
        for line in lines:
            line = line.strip().split(",")
            try:
                label = int(line[4])
                content = line[2]
                vec = self.text2bert(content)
                X.append(vec)
                y.append(label)
            except:
                print(line[0])

        X = np.array(X)
        y = np_utils.to_categorical(np.asarray(y), num_classes=self.nb_classes)
        return X, y

    def data_iter(self):
        """ 数据生成器 """
        fr = codecs.open(self.train_corpus_path, "r", "utf-8")  # 训练集在这里
        lines = fr.readlines()
        fr.close()
        random.shuffle(lines)
        while True:
            for index in range(0, len(lines), self.batch_size):
                batch_samples = lines[index: index+self.batch_size]
                X, y = self.data_format(batch_samples)
                yield (X, y)

    def data_val(self):
        """ 测试数据 """
        fr = codecs.open(self.test_corpus_path, "r", "utf-8")
        lines = fr.readlines()
        fr.close()
        random.shuffle(lines)
        X, y = self.data_format(lines)
        return X,y

    def create_model(self):
        x_in = Input(shape=(768, ))
        x_out = Dense(self.dense_dim, activation="relu")(x_in)
        x_out = BatchNormalization()(x_out)
        x_out = Dense(self.nb_classes, activation="softmax")(x_out)  # 这里是分类
        model = Model(inputs=x_in, outputs=x_out)
        return model

    def train(self):
        model = self.create_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(self.weights_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        x_test, y_test = self.data_val()

        model.fit_generator(self.data_iter(),  # 训练集数据迭代器
                            steps_per_epoch=int(self.nb_samples/self.batch_size)+1,   # batch_size=128
                            epochs=self.epochs,
                            verbose=1,
                            validation_data=(x_test, y_test),
                            validation_steps=None,
                            callbacks=[checkpoint]
                            )
        pred = model.predict(x_test)
        pred = [np.argmax(val) for val in pred]
        print(pred)  # [1, 0, 1, 0, 2, 2, 0, 1, 2, 2, 0, 2, 0, 0, 2, 2, 1]
        y_true = []
        for val in y_test:
            y_true.append(np.argmax(val))
        print(y_true)

        p = precision_score(y_true, pred, average='macro')
        r = recall_score(y_true, pred, average='macro')
        f1 = f1_score(y_true, pred, average='macro')
        print(p)
        print(r)
        print(f1)


if __name__ == "__main__":
    bc = BertClassification()
    bc.train()




