#! -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tqdm import tqdm


num_train = 90000 # 前9万组问题拿来做训练
maxlen = 32
batch_size = 100
min_count = 5
word_size = 128
epochs = 25 # amsoftmax需要25个epoch，其它需要20个epoch


data = pd.read_csv('tongyiju.csv', encoding='utf-8', header=None, delimiter='\t')

def strQ2B(ustring):
    """全角转半角"""
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


from keras.models import Model
from keras.layers import *
from keras.constraints import unit_norm
from margin_softmax import *

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

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs)

model.save_weights('sent_sim_amsoftmax.weights')


def evaluate(num=None):
    """评测函数
    如果按相似度排序后的前n个句子中出现了输入句子的同义句，那么topn的命中数就加1
    """

    if num == None:
        num = len(valid_data)

    print u'测试总数：%s' % num

    valid_vec = encoder.predict(np.array(list(valid_data[2])),
                                verbose=True,
                                batch_size=1000)
    total = 0.
    top1_right = 0.
    top5_right = 0.
    top10_right = 0.
    for k in tqdm(iter(range(num))):
        total += 1
        max_sim_sents = np.dot(valid_vec, valid_vec[k]).argsort()[-11:][::-1]
        max_sim_sents = [valid_data.iloc[i][0] for i in max_sim_sents]
        input_sent = max_sim_sents[0]
        max_sim_sents = max_sim_sents[1:]
        if input_sent == max_sim_sents[0]:
            top1_right += 1
            top5_right += 1
            top10_right += 1
        elif input_sent in max_sim_sents[:5]:
            top5_right += 1
            top10_right += 1
        elif input_sent in max_sim_sents[:10]:
            top10_right += 1
    return top1_right/total, top5_right/total, top10_right/total


print evaluate()
