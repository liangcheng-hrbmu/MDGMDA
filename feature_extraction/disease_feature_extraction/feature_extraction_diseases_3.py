# -*- coding: utf-8 -*-
# @Time     : 2021/12/31 16:07
# @Author   : Ping
# @File     : 特征提取_diseases_1.py
# @Software : PyCharm

import os
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr


def calculate_pointbiserialr(tense_a, tense_b):
    cor_value = pointbiserialr(tense_a, tense_b)[0]
    if np.isnan(cor_value):
        tense_a[0] = tense_a[0] + 0.000001
        cor_value = pointbiserialr(tense_a, tense_b)[0]
    if np.isnan(cor_value):
        tense_b[0] = tense_b[0] + 0.000001
        cor_value = pointbiserialr(tense_a, tense_b)[0]
    return str(round(cor_value, 4))


d_m = pd.read_csv('disease_metabolite.txt', sep='\t', dtype=str).sort_values("DOID", ignore_index=True)

m_m_orig = pd.read_csv('metabolite_metabolite.txt', sep='\t', dtype=str)
m_m_orig["score"] = [eval(x) for x in m_m_orig["score"]]

m_m_copy = pd.DataFrame(m_m_orig[["chemial2", "chemial1", "score"]]).copy()
m_m_copy.columns = ["chemial1", "chemial2", "score"]

m_m = pd.concat([m_m_orig, m_m_copy])

metabolites = sorted(set(m_m["chemial1"]))

with open("disease_list.txt") as fr:
    disease = [x.strip() for x in fr.readlines()]

m_m = pd.concat([m_m, pd.DataFrame(list(zip(metabolites, metabolites, [1] * len(metabolites))),
                                   columns=["chemial1", "chemial2", "score"])]).sort_values(["chemial1", "chemial2"],
                                                                                            ignore_index=True)

tense_mm_dict = {}

metabolite_num = len(metabolites)
for met in metabolites:
    print("%6d of %6d  %s" % ((metabolites.index(met) + 1), metabolite_num, met), end="\r")
    chosen_mm = m_m.loc[m_m["chemial1"] == met].copy()
    chosen_mm_dict = dict(zip(chosen_mm["chemial2"], chosen_mm["score"]))
    tense_mm_dict[met] = [chosen_mm_dict[x] if x in chosen_mm_dict else 0 for x in metabolites]
# species_feature
print()
with open("disease_feature_3.csv.csv", "w") as fw1:
    fw1.write("\t".join(metabolites) + "\n")
    disease_num = len(disease)
    for dis in disease[18:27]:
        print("%4d of %4d  %s" % ((disease.index(dis) + 1), disease_num, dis))
        chosen_d_m = d_m.loc[d_m["DOID"] == dis]["PubChem"].tolist()
        tense_dm = [1 if x in chosen_d_m else 0 for x in metabolites]
        dm_feature = [calculate_pointbiserialr(tense_dm, tense_mm_dict[met]) for met in metabolites]
        fw1.write("\t".join([dis] + dm_feature) + "\n")
