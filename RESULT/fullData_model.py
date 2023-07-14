# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @Time     : 2020/10/19 15:39
  @Author   : Ping Wang
  @Email    :
  @Site     :
  @File     :
  @Software : PyCharm
"""
import os

import pandas as pd
import numpy as np
import random
from keras import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense, Dropout, MaxPool2D
import tensorflow as tf
from keras import regularizers

np.random.seed(7)
tf.random.set_seed(42)
random.seed(14)

os.chdir(r"E:\工作\2020-10-09-5 01_肠道菌-疾病 预测\2021-12-13-特征基于点二列相关\分类model\step2-卷积训练")
# 提取阳性集，构建阴性集
species_feature = pd.read_csv('species_feature.txt', sep='\t', header=0, index_col=0)
disease_feature = pd.read_csv('disease_feature.txt', sep='\t', header=0, index_col=0)

allspecies = list(species_feature.index)
alldiseases = list(disease_feature.index)
d_s = pd.read_csv('disease_species.txt', sep='\t')
all_positive_spe_dis = []
for i in range(d_s.shape[0]):
    if type(d_s['DOID'][i]) is not float:
        if d_s['Gut Microbiota'][i] in allspecies and d_s['DOID'][i] in alldiseases:
            all_positive_spe_dis.append([d_s['Gut Microbiota'][i], d_s['DOID'][i]])
all_spe_dis = []
for spe in allspecies:
    for dis in alldiseases:
        all_spe_dis.append([spe, dis])

positive_spe_dis = [x for x in all_spe_dis if x in all_positive_spe_dis]
negative_spe_dis_all = [x for x in all_spe_dis if x not in all_positive_spe_dis]

negative_spe_dis = random.sample(negative_spe_dis_all, len(positive_spe_dis) * 2)

# evaluate_pos = random.choice(positive_spe_dis)

# positive_spe_dis.remove(positive_spe_dis[0])

spe_dis = positive_spe_dis + negative_spe_dis
label = [1, ] * len(positive_spe_dis) * 2 + [0, ] * len(negative_spe_dis) * 2

SPE_DIS = []
for sd in spe_dis[:20]:
    SPE_DIS.append([list(species_feature.loc[sd[0]]) + [0] * 196,
                    list(disease_feature.loc[sd[1]]) + [0] * 196])
    SPE_DIS.append([list(disease_feature.loc[sd[1]]) + [0] * 196,
                    list(species_feature.loc[sd[0]]) + [0] * 196])

# label = np.array([[1,0] if x==1 else [0,1] for x in label])
label = np.array(label)
SPE_DIS = np.array(SPE_DIS)

SPE_DIS = SPE_DIS.reshape(SPE_DIS.shape[0], -1)

print(SPE_DIS.shape)

# 430 *434
data = SPE_DIS.reshape(SPE_DIS.shape[0], 430, -1, 1)
print(data.shape)


# 卷积模型

# 设置基本参数
def Model():
    basemodel = Sequential()
    basemodel.add(Conv2D(32, (3, 3), padding='same', input_shape=data.shape[1:]))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(MaxPool2D(2, 2))
    basemodel.add(Conv2D(64, (3, 3), padding='same'))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(MaxPool2D(2, 2))
    basemodel.add(Conv2D(128, (3, 3), padding='same'))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(MaxPool2D(2, 2))
    basemodel.add(Flatten())
    # Flatten展平
    basemodel.add(Dense(448, kernel_regularizer=regularizers.l2(0.01)))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(Dense(112))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(Dense(1))
    basemodel.add(Activation('sigmoid'))
    # model.summary()
    # 查看网络结构
    basemodel.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return basemodel  # categorical_crossentropy


model = Model()
model.fit(data, label, batch_size=50, epochs=20)

### 预测

Predict_sd = [x for x in negative_spe_dis_all if x not in negative_spe_dis]

Predict_data = []

for sd in Predict_sd:
    Predict_data.append([list(species_feature.loc[sd[0]]) + [0] * 196,
                         list(disease_feature.loc[sd[1]]) + [0] * 196])
Predict_data = np.array(Predict_data)
print(Predict_data.shape)
Predict_data = Predict_data.reshape(Predict_data.shape[0], 430, -1, 1)
print(Predict_data.shape)
Predict_result = model.predict(Predict_data)
Predict_result = [x[0] for x in Predict_result]

predictionResult = pd.DataFrame(columns=['species', 'disease', 'correlation'])

species = [x[0] for x in Predict_sd]
disease = [x[1] for x in Predict_sd]

predictionResult['species'] = species

predictionResult['disease'] = disease

predictionResult['correlation'] = Predict_result

predictionResult = predictionResult.sort_values('correlation', ascending=False)

predictionResult.to_csv('fullDataForTrain_PredictionResult.csv', index=None, sep='\t')
