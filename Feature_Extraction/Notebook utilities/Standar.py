import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import tqdm 
import os 
from os.path import dirname, join as pjoin
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import scipy.io as sio
import copy

def Standarize_FC(filename):
    sc = StandardScaler()

    FC = sio.loadmat(filename)
    n_subj = len(FC['img'][0])
    len_data = len(FC['img'][0][0][1])

    fc_3d = np.zeros((n_subj,len_data, len_data) )

    for subject in tqdm(range(n_subj)):

        if len(FC['img'][0][subject][1]) != 0:
            if np.all(FC['img'][0][subject][1] != 0):
                df = pd.DataFrame(FC['img'][0][subject][1])
                #converting NA to 0 values - The one in the diagonal are NA
                df = df.fillna(0)
                sc.fit(df)
                df = sc.transform(df)

                fc_3d[subject] = np.asanyarray(df)
                
                
    idx = 0
    indexes = []
    for i in fc_3d:
        if np.all(i == 0):
            indexes.append(idx)
        idx+=1
    fc_3d = np.delete(fc_3d, indexes, axis = 0)
    
    return fc_3d


def data_cleansing(language_score, features):
    scores = copy.deepcopy(language_score)
    indeces = []
    for idx in range(len(language_score)):
        if (language_score[idx] == '[]'):
            indeces.append(idx)
            
    for idx in indeces[::-1]:
        scores.pop(idx)
    for key in features.keys():
        for idx in indeces[::-1]:
            features[key] = np.delete(features[key], idx, 0)
            
    scores = (scores - np.mean(scores) )/np.std(scores, ddof=1)
    
    return features, scores