# -*- coding: utf-8 -*- 
"""
Project         : 2020-10-09-5 01_肠道菌-疾病 预测
File            : fullData_model_LOO_2.py
Creator         : Ping
Create time     : 2022/9/28 8:36
Software        : PyCharm
Introduction    :
"""

import random

import brewer2mpl
import numpy as np
import pandas as pd
# 5-折交叉验证
from sklearn.model_selection import LeaveOneOut
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, Dropout, MaxPool2D
from tqdm import tqdm
from tqdm import trange

bmap = brewer2mpl.get_map('dark2', 'qualitative', 7)
colors = bmap.mpl_colors

# 分别读取 微生物、 疾病特征
print("Loading the feature matrix")
species_feature = pd.read_csv('species_feature.txt.gz', sep='\t', header=0, index_col=0, compression='gzip')
disease_feature = pd.read_csv('disease_feature.txt.gz', sep='\t', header=0, index_col=0, compression='gzip')

print("species_feature:", species_feature.shape)
print("disease_feature:", disease_feature.shape)

# 构建 微生物-疾病关系对， 分为阳性集、阴性集。
print("Construct the Positive set and negative set")
allspecies = list(species_feature.index)
alldiseases = list(disease_feature.index)
d_s = pd.read_csv('random0/disease_species.txt', sep='\t')
all_positive_spe_dis = []
for i in trange(d_s.shape[0]):
    if type(d_s['DOID'][i]) is not float:
        if d_s['Gut Microbiota'][i] in allspecies and d_s['DOID'][i] in alldiseases:
            all_positive_spe_dis.append([d_s['Gut Microbiota'][i], d_s['DOID'][i]])
all_spe_dis = []
for spe in tqdm(allspecies):
    for dis in alldiseases:
        all_spe_dis.append([spe, dis])

positive_spe_dis = [x for x in all_spe_dis if x in all_positive_spe_dis]
negative_spe_dis_all = [x for x in all_spe_dis if x not in all_positive_spe_dis]
negative_spe_dis = random.sample(negative_spe_dis_all, len(positive_spe_dis) * 2)

# 构建模型特征数据集及标签
spe_dis = positive_spe_dis + negative_spe_dis
label = [1] * len(positive_spe_dis) * 2 + [0] * len(negative_spe_dis) * 2
SPE_DIS = []
for sd in tqdm(spe_dis):
    SPE_DIS.append([list(species_feature.loc[sd[0]]),
                    list(disease_feature.loc[sd[1]])])
    SPE_DIS.append([list(disease_feature.loc[sd[1]]),
                    list(species_feature.loc[sd[0]])])
# label = np.array([[1,0] if x==1 else [0,1] for x in label])
label = np.array(label)
SPE_DIS = np.array(SPE_DIS)
SPE_DIS = SPE_DIS.reshape(SPE_DIS.shape[0], -1)
print(SPE_DIS.shape)
# 430 *434
data = SPE_DIS.reshape(SPE_DIS.shape[0], 252, -1, 1)
print(data.shape)


# 卷积模型

# 设置基本参数
# 搭建模型框架
def Model():
    basemodel = Sequential()
    basemodel.add(Conv2D(32, (3, 3), padding='same', input_shape=data.shape[1:]))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(MaxPool2D(3, 3))
    basemodel.add(Conv2D(64, (3, 3), padding='same'))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(MaxPool2D(3, 3))
    basemodel.add(Conv2D(128, (3, 3), padding='same'))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(MaxPool2D(3, 3))
    basemodel.add(Flatten())
    # Flatten展平
    basemodel.add(Dense(640, kernel_regularizer=regularizers.l2(0.01)))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(Dense(320))
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

### LOO
print("Start leave one out model")
loo = LeaveOneOut()

with open("LOO_result_191_1000.csv", "w") as fw:
    loop_num = loo.get_n_splits(data)
    loop = 0
    for train_index, test_index in loo.split(data):
        loop += 1
        if loop <= 190:
            continue
        if loop > 1000:
            break
        tr_x, tr_y = data[train_index], label[train_index]
        te_x, te_y = data[test_index], label[test_index]
        model = Model()
        model.fit(tr_x, tr_y, batch_size=50, epochs=20, verbose=0)
        pred = model.predict(te_x)[0].tolist()[0]
        result = "%s\t%s\t%s\n" % (loop, pred, te_y.tolist()[0])
        print(loop, result, end=" - ")
        fw.write(result)
        fw.flush()


# fig = plt.figure(figsize=(10, 10))
# # plt.style.use('fivethirtyeight')
# plt.style.use('default')
# plt.rc('axes', prop_cycle=(cycler('color', colors)))
# plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=0.5, color='grey')
#
# fpr, tpr, thresholds = roc_curve(reallabel_list, predictlabel_list, pos_label=1)
# auc_score = auc(fpr, tpr)
# plt.plot(fpr, tpr, lw=1, alpha=0.5, label='ROC LOO (area=%0.2f)' % auc_score)
# plt.xlim([-0.005, 1.005])
# plt.ylim([-0.005, 1.005])
# plt.xlabel('False Positive Rate', )
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.legend(loc='lower right')
# plt.show()
# fig.savefig("5-fold_orig.pdf")
