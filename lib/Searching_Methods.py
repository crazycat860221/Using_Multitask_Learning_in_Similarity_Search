import numpy as np
from scipy.spatial import distance
from scipy import stats
from time import time

from lib.Load import Load_file
from lib.Compute_time import Compute_time
from args import parser

args = parser()

# createIndex using Two Stage Zscore p_idx
def createIndex(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, p_prob, Zscore_var, testcases, FC, SC, model):
    FNormalize = Load_file(args.Train_Loc+args.Residual_Train_Path)
    NNormalize = np.max(FNormalize)

    FcandidateSize = FC
    ScandidateSize = SC
    DScandidateSize = np.zeros((query_sift.shape[0], FC), dtype=int)
    BucketSize = FcandidateSize *  ScandidateSize

    CNN = np.zeros((query_sift.shape[0], FcandidateSize), dtype=int)
    FCandidate = np.zeros((FcandidateSize, FirstStage_Codebook.shape[1]), dtype=float)

    InsIdx2 = np.zeros((query_sift.shape[0], FcandidateSize *  ScandidateSize), dtype=int)
    InsIdx2 -= 1

    start_time = time()

    for i in range (query_sift.shape[0]):
        idx = learning_idx[i]
        CNN[i,0:FcandidateSize] = idx[0:FcandidateSize]

    #===========Test Outlier
        SortedDist = p_prob[i,idx[0:Zscore_var]]
        ZScoreSortedList = stats.zscore(SortedDist)

        invZScore = ZScoreSortedList

        sumZScore = sum(x for x in invZScore if x > 0)
        probRL1 = invZScore[invZScore>0]
        probRL2 = probRL1/sumZScore  #Probability of how many data we want to retrieve from this cluster
        MaxProbRL2 = np.argmax(probRL2)

        len_probRL2 = len(probRL2)
        excess = 0
        for y in range(len_probRL2):
            if y < FC:
                DScandidateSize[i,y] = int(probRL2[y]*(FC*SC)) + excess
                if DScandidateSize[i,y] > SecondStage_Codebook.shape[0]:
                    excess = DScandidateSize[i,y] -  (SecondStage_Codebook.shape[0]*(1-(y/10)))
                    DScandidateSize[i,y] = SecondStage_Codebook.shape[0] * (1-(y/10))
                else:
                    excess = 0

            else :
                if MaxProbRL2 < FcandidateSize:
                    DScandidateSize[i,MaxProbRL2] += int(probRL2[y]*(FC*SC))
                else:
                    DScandidateSize[i,0] += int(probRL2[y]*(FC*SC))

        if i in testcases:
            print("++++++++++++++++++++++++++++++++++++++++")
            print(i)
            print("First Stage cluster------------")
            print(CNN[i])
            print(DScandidateSize[i])

        DRCNN = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize ), dtype=int)
        DRCNN -= 1
        DSorted_Rdist = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize), dtype=float)
        DSorted_Rdist -=1
        if len_probRL2 < FcandidateSize:
            RANGE = len_probRL2

        else:
            RANGE = FcandidateSize

        x_test = np.zeros((FcandidateSize,SecondStage_Codebook.shape[1]*2), dtype=float)

        for j in range(RANGE):

           FCandidate[j] = query_sift[i] - FirstStage_Codebook[CNN[i,j]]
           x_test[j,0:SecondStage_Codebook.shape[1]] = FirstStage_Codebook[CNN[i,j]]
           x_test[j,SecondStage_Codebook.shape[1]:SecondStage_Codebook.shape[1]*2] = FCandidate[j]
           DataSize = DScandidateSize[i,j]

        x_test /= NNormalize

        p_test = model(x_test)
        p_idx = np.argsort(-p_test)

        for j in range(FcandidateSize):
            DataSize = DScandidateSize[i,j]
            DRCNN[j, 0:DataSize] = p_idx[j,0:DataSize]


        temp_candidate = []
        candidate = []

        Alternate = 3
        if FcandidateSize > 1:
            for r in range (Alternate):
                temp_candidate = (CNN[i,0]*SecondStage_Codebook.shape[0])+(DRCNN[0,r])
                candidate.extend([temp_candidate])
                temp_candidate = (CNN[i,1]*SecondStage_Codebook.shape[0])+(DRCNN[1,r])
                candidate.extend([temp_candidate])

            for j in range(RANGE):
                if j < 2:
                    start = Alternate
                else:
                    start = 0

                for k in range(start, DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k])
                    candidate.extend([temp_candidate])
        else:
                for k in range(DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k])
                    candidate.extend([temp_candidate])

        CandidateList = np.array(candidate)
        IdxSize = len(CandidateList)
        if IdxSize > BucketSize:
            IdxSize = BucketSize
        InsIdx2[i,0:IdxSize] =  CandidateList[0:IdxSize]

    end_time = time()
    # Compute_time(start_time, end_time, 'Create Index Time')
    
    return InsIdx2


# createIndex using Two Stage Zscore p_idx APPLY EARLY TERMINATION
def createIndex_EarlyTermination(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, p_prob, Zscore_var, testcases, FC, SC, model, p_ter):
    FNormalize = Load_file(args.Train_Loc+args.Residual_Train_Path)
    NNormalize = np.max(FNormalize)

    start_time = time()
    FcandidateSize = FC
    ScandidateSize = SC
    DScandidateSize = np.zeros((query_sift.shape[0], FC), dtype=int)
    BucketSize = FcandidateSize *  ScandidateSize

    CNN = np.zeros((query_sift.shape[0], FcandidateSize), dtype=int)
    FCandidate = np.zeros((FcandidateSize, FirstStage_Codebook.shape[1]), dtype=float)

    InsIdx2 = np.zeros((query_sift.shape[0], FcandidateSize *  ScandidateSize), dtype=int)
    InsIdx2 -= 1

    for i in range (query_sift.shape[0]):
        FC = int(p_ter[i])
        idx = learning_idx[i]
        CNN[i,0:FC] = idx[0:FC]

    #===========Test Outlier
        SortedDist = p_prob[i,idx[0:Zscore_var]]
        ZScoreSortedList = stats.zscore(SortedDist)

        invZScore = ZScoreSortedList

        sumZScore = sum(x for x in invZScore if x > 0)
        probRL1 = invZScore[invZScore>0]
        probRL2 = probRL1/sumZScore  #Probability of how many data we want to retrieve from this cluster
        MaxProbRL2 = np.argmax(probRL2)

        len_probRL2 = len(probRL2)
        excess = 0
        for y in range(len_probRL2):
            if y < FC:
                DScandidateSize[i,y] = int(probRL2[y]*(FC*SC)) + excess
                if DScandidateSize[i,y] > SecondStage_Codebook.shape[0]:
                    excess = DScandidateSize[i,y] -  (SecondStage_Codebook.shape[0]*(1-(y/10)))
                    DScandidateSize[i,y] = SecondStage_Codebook.shape[0] * (1-(y/10))
                else:
                    excess = 0

            else :
                if MaxProbRL2 < FC:
                    DScandidateSize[i,MaxProbRL2] += int(probRL2[y]*(FC*SC))
                else:
                    DScandidateSize[i,0] += int(probRL2[y]*(FC*SC))

        if i in testcases:
            print("++++++++++++++++++++++++++++++++++++++++")
            print(i)
            print("First Stage cluster------------")
            print(CNN[i])
            print(DScandidateSize[i])

        DRCNN = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize ), dtype=int)
        DRCNN -= 1
        DSorted_Rdist = np.zeros((FcandidateSize,FcandidateSize * ScandidateSize), dtype=float)
        DSorted_Rdist -=1
        if len_probRL2 < FC:
            RANGE = len_probRL2

        else:
            RANGE = FC

        x_test = np.zeros((FC,SecondStage_Codebook.shape[1]*2), dtype=float)

        for j in range(RANGE):

           FCandidate[j] = query_sift[i] - FirstStage_Codebook[CNN[i,j]]
           x_test[j,0:SecondStage_Codebook.shape[1]] = FirstStage_Codebook[CNN[i,j]]
           x_test[j,SecondStage_Codebook.shape[1]:SecondStage_Codebook.shape[1]*2] = FCandidate[j]
           DataSize = DScandidateSize[i,j]

        x_test /= NNormalize

        p_test = model(x_test)
        p_idx = np.argsort(-p_test)

        for j in range(FC):
            DataSize = DScandidateSize[i,j]
            DRCNN[j, 0:DataSize] = p_idx[j,0:DataSize]


        temp_candidate = []
        candidate = []

        Alternate = 3
        if FC > 1:
            for r in range (Alternate):
                temp_candidate = (CNN[i,0]*SecondStage_Codebook.shape[0])+(DRCNN[0,r])
                candidate.extend([temp_candidate])
                temp_candidate = (CNN[i,1]*SecondStage_Codebook.shape[0])+(DRCNN[1,r])
                candidate.extend([temp_candidate])

            for j in range(RANGE):
                if j < 2:
                    start = Alternate
                else:
                    start = 0

                for k in range(start, DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k])
                    candidate.extend([temp_candidate])
        else:
                for k in range(DScandidateSize[i,j]):
                    temp_candidate = (CNN[i,j]*SecondStage_Codebook.shape[0])+(DRCNN[j,k])
                    candidate.extend([temp_candidate])

        CandidateList = np.array(candidate)
        IdxSize = len(CandidateList)
        if IdxSize > BucketSize:
            IdxSize = BucketSize
        InsIdx2[i,0:IdxSize] =  CandidateList[0:IdxSize]

    end_time = time()
    # Compute_time(start_time, end_time, 'Create Index Time With EARLY TERMINATION')

    return InsIdx2


