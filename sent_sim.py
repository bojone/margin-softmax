#! -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from keras.models import Model
from keras.layers import *
from keras.constraints import unit_norm
from margin_softmax import *
from keras.callbacks import Callback


num_train = 90000 # 前9万组问题拿来做训练
maxlen = 32
batch_size = 100
min_count = 5
word_size = 128
epochs = 30 # amsoftmax需要25个epoch，其它需要20个epoch


data = pd.read_csv('tongyiju.csv', encoding='utf-8', header=None, delimiter='\t')

def strQ2B(ustring): # 全角转半角
    rstring = ''
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288: # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring


data[1] = data[1].apply(strQ2B)
data[1] = data[1].str.lower()

chars = {}
for s in tqdm(iter(data[1])):
    for c in s:
        if c not in chars:
            chars[c] = 0
        chars[c] += 1


# 0: padding标记
# 1: unk标记
chars = {i:j for i,j in chars.items() if j >= min_count}
id2char = {i+2:j for i,j in enumerate(chars)}
char2id = {j:i for i,j in id2char.items()}

def string2id(s):
    _ = [char2id.get(i, 1) for i in s[:maxlen]]
    _ = _ + [0] * (maxlen - len(_))
    return _


data[2] = data[1].apply(string2id)
train_data = data[data[0] < num_train]
train_data = train_data.sample(frac=1)
x_train = np.array(list(train_data[2]))
y_train = np.array(list(train_data[0])).reshape((-1,1))

valid_data = data[data[0] >= num_train]


# 正式模型，基于GRU的分类器
x_in = Input(shape=(maxlen,))
x_embedded = Embedding(len(chars)+2,
                       word_size)(x_in)
x = CuDNNGRU(word_size)(x_embedded)
x = Lambda(lambda x: K.l2_normalize(x, 1))(x)

pred = Dense(num_train,
             use_bias=False,
             kernel_constraint=unit_norm())(x)

encoder = Model(x_in, x) # 最终的目的是要得到一个编码器
model = Model(x_in, pred) # 用分类问题做训练

model.compile(loss=sparse_amsoftmax_loss,
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])


# 为验证集的排序准备
# 实际上用numpy写也没有问题，但是用Keras写能借助GPU加速
x_in = Input(shape=(word_size,))
x = Dense(len(valid_data), use_bias=False)(x_in) # 计算相似度
x = Lambda(lambda x: K.tf.nn.top_k(x, 11)[1])(x) # 取出topk的下标
model_sort = Model(x_in, x)

# id与组别之间的映射
id2g = dict(zip(valid_data.index-valid_data.index[0], valid_data[0]))


def evaluate(): # 评测函数
    print 'validing...'
    valid_vec = encoder.predict(np.array(list(valid_data[2])),
                                verbose=True,
                                batch_size=1000) # encoder计算句向量
    model_sort.set_weights([valid_vec.T]) # 载入句向量为权重
    sorted_result = model_sort.predict(valid_vec,
                                       verbose=True,
                                       batch_size=1000) # 计算topk
    new_result = np.vectorize(lambda s: id2g[s])(sorted_result)
    _ = new_result[:, 0] != new_result[:, 0] # 生成一个全为False的向量
    for i in range(10): # 注意按照相似度排序的话，第一个就是输入句子（全匹配）
        _ = _ + (new_result[:, 0] == new_result[:, i+1])
        if i+1 == 1:
            top1_acc = 1. * _.sum() / len(_)
        elif i+1 == 5:
            top5_acc = 1. * _.sum() / len(_)
        elif i+1 == 10:
            top10_acc = 1. * _.sum() / len(_)

    return top1_acc, top5_acc, top10_acc


# 定义Callback器，计算验证集的acc，并保存最优模型
class Evaluate(Callback):
    def __init__(self):
        self.top1_accs = []
        self.highest = 0.
    def on_epoch_end(self, epoch, logs=None):
        top1_acc, top5_acc, top10_acc = evaluate()
        self.top1_accs.append(top1_acc)
        if top1_acc >= self.highest: # 保存最优模型权重
            self.highest = top1_acc
            model.save_weights('sent_sim_amsoftmax.model')
        json.dump([self.scores, self.highest], open('valid.log', 'w'), indent=4)
        print 'top1_acc: %s, top5_acc: %s, top10_acc: %s' % (top1_acc, top5_acc, top10_acc)


evaluator = Evaluate()

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[evaluator])


valid_vec = encoder.predict(np.array(list(valid_data[2])),
                            verbose=True,
                            batch_size=1000) # encoder计算句向量

def most_similar(s):
    v = encoder.predict(np.array([string2id(s)]))[0]
    sims = np.dot(valid_vec, v)
    for i in sims.argsort()[-10:][::-1]:
        print valid_data.iloc[i][1],sims[i]


most_similar(u'ps格式可以转换成ai格式吗')
most_similar(u'广州的客运站的数目')
most_similar(u'沙发一般有多高')
