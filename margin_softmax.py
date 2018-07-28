#! -*- coding: utf-8 -*-


# 普通sparse交叉熵，以logits为输入
def sparse_logits_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
