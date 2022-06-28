from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from lib import First_Level_Prediction
from lib import TwoStage_Searching

from args import parser

if __name__ == '__main__':

    MaxFirstStageCluster = 15
    NumSubCluster = 10
    Zscore_var =  100
    START = 15
    STEP = 1

    #== Load data
    print('\nLoading input files')
    
    npzfile = np.load('./input/Idx2Codebooks.npz', allow_pickle=True)
    Ints_clustMem = np.array(npzfile['idx'])
    npzfile = np.load('./input/Idx2CountMember.npz')
    idxCount = np.array(npzfile['idxCount'])
    Top1 = np.load('./input/Test_Data/Top1_Test.npy').astype(int)

    FirstStage_Codebook = np.load('./input/FirstStageCodebook.npy')
    SecondStage_Codebook = np.load('./input/SecondStageCodebook.npy')

    #== First level predict
    First_Level_Prediction.First_Predict()    
    
    #== Two stage retrieval
    TwoStage_Searching.Searching(MaxFirstStageCluster, NumSubCluster, Zscore_var, 
                                    Ints_clustMem, idxCount, Top1, FirstStage_Codebook, SecondStage_Codebook, START, STEP)
                
    TwoStage_Searching.SearchingEarlyTer(MaxFirstStageCluster, NumSubCluster, Zscore_var, 
                                            Ints_clustMem, idxCount, Top1, FirstStage_Codebook, SecondStage_Codebook, START, STEP)



