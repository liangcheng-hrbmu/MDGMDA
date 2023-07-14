# -*- coding: utf-8 -*- 
"""
Project         :
File            : clusterByPhylum_models.py
Creator         : Ping
Create time     : 2022/11/8 15:51
Software        : PyCharm
Introduction    :
"""

import random
import os
import brewer2mpl
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, Dropout, MaxPool2D, GaussianNoise
from tqdm import tqdm
from tqdm import trange


bmap = brewer2mpl.get_map('dark2', 'qualitative', 7)
colors = bmap.mpl_colors

species_feature = pd.read_csv('species_feature.txt.gz', sep='\t', header=0, index_col=0, compression='gzip')
disease_feature = pd.read_csv('disease_feature.txt.gz', sep='\t', header=0, index_col=0, compression='gzip')
print("species_feature:", species_feature.shape)
print("disease_feature:", disease_feature.shape)

d_s = pd.read_csv('disease_species.txt', sep='\t')

allspecies = list(species_feature.index)
alldiseases = list(disease_feature.index)
all_positive_spe_dis = []
for i in trange(d_s.shape[0]):
    if type(d_s['DOID'][i]) is not float:
        if d_s['Gut Microbiota'][i] in allspecies and d_s['DOID'][i] in alldiseases:
            all_positive_spe_dis.append([d_s['Gut Microbiota'][i], d_s['DOID'][i]])
all_spe_dis = []
for spe in tqdm(allspecies):
    for dis in alldiseases:
        all_spe_dis.append([spe, dis])

positive_spe_dis_all = [x for x in all_spe_dis if x in all_positive_spe_dis]

negative_spe_dis_all = [x for x in all_spe_dis if x not in all_positive_spe_dis]


def construct_dataset(positive_spe_dis_):

    negative_spe_dis = random.sample(negative_spe_dis_all, len(positive_spe_dis_) * 2)

    spe_dis = positive_spe_dis_ + negative_spe_dis
    label = [1] * len(positive_spe_dis_) * 2 + [0] * len(negative_spe_dis) * 2
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
    return data, label


def Model_gauss(noise=0.01):
    basemodel = Sequential()
    basemodel.add(Conv2D(32, (5, 5), padding='same', input_shape=data.shape[1:]))
    if noise != 0:
        basemodel.add(GaussianNoise(noise))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(MaxPool2D(5, 5))

    basemodel.add(Conv2D(64, (5, 5), padding='same'))
    # basemodel.add(GaussianNoise(0.1))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(MaxPool2D(5, 5))

    basemodel.add(Conv2D(128, (5, 5), padding='same'))
    # basemodel.add(GaussianNoise(0.1))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(MaxPool2D(5, 5))
    basemodel.add(Flatten())

    # Flatten
    basemodel.add(Dense(640, kernel_regularizer=regularizers.l2(0.01)))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(Dense(320))
    basemodel.add(Activation('relu'))
    basemodel.add(Dropout(0.1))
    basemodel.add(Dense(1))
    basemodel.add(Activation('sigmoid'))
    # model.summary()
    basemodel.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return basemodel  # categorical_crossentropy


phylum_classification = pd.read_excel("微生物分类.xlsx")

phylum_classification_dict = {}

for value in phylum_classification.values:
    if value[1] not in phylum_classification_dict:
        phylum_classification_dict[value[1]] = [value[0]]
    else:
        phylum_classification_dict[value[1]].append(value[0])
if not os.path.exists("ClusterByPhylum"):
    os.mkdir("ClusterByPhylum")
for phy in phylum_classification_dict:
    phy_list = phylum_classification_dict[phy]
    if not os.path.exists("ClusterByPhylum/%s" % phy):
        os.mkdir("ClusterByPhylum/%s" % phy)
    positive_spe_dis = [x for x in positive_spe_dis_all if x[0] in phy_list]
    ###

    noise = 0
    for circ in tqdm(range(0, 20)):
        data, label = construct_dataset(positive_spe_dis)
        print(noise, circ+1)
        KF = StratifiedKFold(n_splits=5, shuffle=True)
        AUCs = []
        Labels = []
        Predict_result = []
        loop = tqdm(KF.split(data, label), total=5)
        for train, test in loop:
            tr_x, tr_y = data[train], label[train]
            te_x, te_y = data[test], label[test]
            model = Model_gauss(noise=noise)
            model.fit(tr_x, tr_y, batch_size=50, epochs=20, verbose=0)
            pred = model.predict(te_x)
            # fpr, tpr, thresholds = roc_curve(te_y, pred, pos_label=1)
            # auc_score = auc(fpr, tpr)
            # print(auc_score)
            # AUCs.append(auc_score)
            Predict_result.append(pred)
            Labels.append(te_y)
        i = 1
        with open("ClusterByPhylum/%s/5-fold_gauss_%s_result_%s.csv" %
                  (phy, noise, (circ + 1)), "w") as fw:
            fw.write(str([x.reshape(-1).tolist() for x in Predict_result]) + "\n" +
                     str([x.tolist() for x in Labels]))
