#! -*- coding: utf-8 -*-

import keras.backend as K


# 普通sparse交叉熵，以logits为输入
def sparse_logits_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


# 稀疏版AM-Softmax
def sparse_amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_true = K.expand_dims(y_true[:, 0], 1) # 保证y_true的shape=(None, 1)
    y_true = K.cast(y_true, 'int32') # 保证y_true的dtype=int32
    batch_idxs = K.arange(0, K.shape(y_true)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, y_true], 1)
    y_true_pred = K.tf.gather_nd(y_pred, idxs) # 目标特征，用tf.gather_nd提取出来
    y_true_pred = K.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - margin # 减去margin
    _Z = K.concatenate([y_pred, y_true_pred_margin], 1) # 为计算配分函数
    _Z = _Z * scale # 缩放结果，主要因为pred是cos值，范围[-1, 1]
    logZ = K.logsumexp(_Z, 1, keepdims=True) # 用logsumexp，保证梯度不消失
    logZ = logZ + K.log(1 - K.exp(scale * y_true_pred - logZ)) # 从Z中减去exp(scale * y_true_pred)
    return - y_true_pred_margin * scale + logZ


# 简单的类A-Softmax（m=4）
def sparse_simpler_asoftmax_loss(y_true, y_pred, scale=30):
    y_true = K.expand_dims(y_true[:, 0], 1) # 保证y_true的shape=(None, 1)
    y_true = K.cast(y_true, 'int32') # 保证y_true的dtype=int32
    batch_idxs = K.arange(0, K.shape(y_true)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, y_true], 1)
    y_true_pred = K.tf.gather_nd(y_pred, idxs) # 目标特征，用tf.gather_nd提取出来
    y_true_pred = K.expand_dims(y_true_pred, 1)
    # 用到了四倍角公式进行展开
    y_true_pred_margin = 1 - 8 * K.square(y_true_pred) + 8 * K.square(K.square(y_true_pred))
    # 下面等效于min(y_true_pred, y_true_pred_margin)
    y_true_pred_margin = y_true_pred_margin - K.relu(y_true_pred_margin - y_true_pred)
    _Z = K.concatenate([y_pred, y_true_pred_margin], 1) # 为计算配分函数
    _Z = _Z * scale # 缩放结果，主要因为pred是cos值，范围[-1, 1]
    logZ = K.logsumexp(_Z, 1, keepdims=True) # 用logsumexp，保证梯度不消失
    logZ = logZ + K.log(1 - K.exp(scale * y_true_pred - logZ)) # 从Z中减去exp(scale * y_true_pred)
    return - y_true_pred_margin * scale + logZ
