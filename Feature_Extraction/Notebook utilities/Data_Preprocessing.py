import scipy.io as sio
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd # this module is useful to work with tabular data
from collections.abc import Iterable

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

def NormalizeData(data):
    "Function for normalizing the fc matrices between 0 and 1"
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def StandarizePandas(df):
    "Function for standarizing a pandas matrices with mean 0 and std 1"
    mean = np.mean([df[col].mean() for col in df.columns], axis=0)
    std = np.std([df[col].std() for col in df.columns], axis=0)
    return mean, std

def get_arrays(mat_path, score_path, Normalize):
    """
    Function for obtaining the arrays of the stroke dataset from the .mat file
    mat_path : path to the .mat file containing the stroke dataset
    score_path: path to the .xsl file containing the scores of each of the patient
    It returns a 3d array of the matrices and an array with the scores per each of them (only THE MEANINGFUL DATA - THe NAN values are removed)
    Normalize == True it returns the fc matrices normalized, otherwise they are standarize (mean = 0 and std = 1)
    """
    
    #Importing the data from the .mat file
    FC = sio.loadmat(mat_path)
    
    #importing the scores from the .xls file 
    FC_language = pd.read_excel(score_path,  converters={'Subj ID':str}, convert_float=False, engine='openpyxl')
        
    #matrix containing only the fc matrices
    fc_stroke = []
    
    #contiaing the ID for each of them
    ID_stroke = []
    
    #containing the language score for each of them
    lang_score = []
    
    nsubj = len(FC['img'][0])
    
    for i in range(nsubj):
        
        #only keep the relevant matrices: - some data has zero values -
        if (len(FC['img'][0][i][1]) != 0):
          
            #generating dataframe containing only the fc matrix
            df = pd.DataFrame(FC['img'][0][i][1])
            
            #converting NA to 0 values - The one in the diagonal are NA
            df = df.fillna(0)

            #append matrices 
            if Normalize == True:
                fc_stroke.append(NormalizeData(np.array(df)))
            else:
                df = np.asanyarray(df)
                fc_stroke.append(np.array(df))
                
            #get ID
            ID_stroke.append(FC['img'][0][i][0][0])

            #match matrix with ID and score
            f = FC_language[FC_language['Subj ID'] == FC['img'][0][i][0][0]]
            try:
                val = f['Language']
                lang_score.append(val.values[0])
            except:
                lang_score.append(np.array([0]))

    
    #generate 3D matrix of the fc for the data that is ok
    nsubj_ok = len(lang_score)
    mat_size = len(fc_stroke[0])
    
    fc_3d = np.zeros((nsubj_ok, mat_size,mat_size))
    
    for i in range(nsubj_ok):
        for j in range(mat_size):
            for k in range(mat_size):
                fc_3d[i][j][k] = fc_stroke[i][j][k]
        
    return fc_3d, lang_score, ID_stroke


def flatten(lis):
    #Function for flatten a list of list and simply convert it to a list
     for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                yield x
        else:        
             yield item
                
                
def to_vector(fc):
    "Function that transform the upper part of the matrix in a vector form"
    
    n_subj = len(fc)
    dim_data = int((len(fc[0])**2 - len(fc[0]))/2)
    
    data = np.zeros((n_subj, dim_data))
    
    for i in range(n_subj):
        vec = []
        
        for row in range(len(fc[0])-1):
            line = fc[i][row][(row+1):len(fc[0])]
            sq = np.squeeze(line)
            vec.append(sq.tolist())
        
        vect_matrices = list(flatten(vec))
        data[i] = vect_matrices
        
    return data


def to_matrix(vector):
    ''''''''''
    Function that convert the vector to the original matrix version
    '''''''''
    dim = 324
    #notice that the vector represent the upper_triangular part of a matrix wich is symmetric
    upper_triangular = np.zeros((dim,dim))
    upper_triangular[np.triu_indices(dim, 1)] = vector
    
    #computing the lower part
    for i in range(len(upper_triangular)):
        for j in range(len(upper_triangular)):
            upper_triangular[j][i] = upper_triangular[i][j]
            
    return upper_triangular


def get_HCP(files):
    
    FC = sio.loadmat(files[0])
    n_subj = len(files)
    len_data = len(FC['FC1'])

    FC_3D = np.zeros((n_subj,len_data, len_data) )
    for subject in tqdm(range(n_subj)):
        FC = sio.loadmat(files[subject])
        if len(FC['FC1']) != 0:
            df = pd.DataFrame(FC['FC1'])
            #converting NA to 0 values - The one in the diagonal are NA
            df = df.fillna(0)
            FC_3D[subject] =  np.asanyarray(df)
            
            
    #removing matrices full of zeros
    idx = 0
    indexes = []
    for i in FC_3D:
        if np.all(i == 0):
            indexes.append(idx)

        idx+=1
    FC_3D = np.delete(FC_3D, indexes, axis = 0)
    return FC_3D