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


# species_metabolite
s_m = pd.read_csv('species_metabolites.csv', dtype=str).sort_values("species", ignore_index=True)

m_m_orig = pd.read_csv('metabolite_metabolite.txt', sep='\t', dtype=str)
m_m_orig["score"] = [eval(x) for x in m_m_orig["score"]]

m_m_copy = pd.DataFrame(m_m_orig[["chemial2", "chemial1", "score"]]).copy()
m_m_copy.columns = ["chemial1", "chemial2", "score"]

m_m = pd.concat([m_m_orig, m_m_copy])

metabolites = sorted(set(m_m["chemial1"]))

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

with open("species_list.txt") as fr:
    species = [x.strip() for x in fr.readlines()]

print()
with open("species_feature_2.csv", "w") as fw1:
    species_num = len(species)
    for spe in species[13:26]:
        print("%4d of %4d  %s" % ((species.index(spe) + 1), species_num, spe))
        chosen_s_m = s_m.loc[s_m["species"] == spe]["PubChem"].tolist()
        tense_sm = [1 if x in chosen_s_m else 0 for x in metabolites]
        sm_feature = [calculate_pointbiserialr(tense_sm, tense_mm_dict[met]) for met in metabolites]
        fw1.write("\t".join([spe] + sm_feature) + "\n")
