import numpy as np
from time import time

from lib.Network import ResidualModel
from lib.Searching_Methods import createIndex, createIndex_EarlyTermination
from lib.Evaluating import Searching_Top1
from lib.Load import Load_file
from args import parser


args = parser()

# Fixed configuration
def Searching(MaxFC, NumSubCluster, Zscore_var, Ints_clustMem, idxCount, Top1, FirstStage_Codebook, SecondStage_Codebook, START, STEP):
    print('\nSearching with fixed configuration\n')   
    model = ResidualModel()
    model.load_weights(args.ResidualModel_Weight_Path)
    
    query_sift = Load_file(args.Test_Loc+args.Test_Data_Path)
    learning_idx = Load_file(args.Output_Loc+args.First_pIdx_Path).astype(int)
    p_prob = Load_file(args.Output_Loc+args.First_pProb_Path)
    
    Method = 1
    testcases = []
    
    CTop1_Recall  = np.zeros((MaxFC, Method), dtype=float)
    CTop1_MeanCandidate  = np.zeros((MaxFC, Method), dtype=int)
    
    cluster_idx = 0 
    for FC in range (START, MaxFC+1, STEP):
        
        Top1Recall  = np.zeros((MaxFC, Method), dtype=float)
        Top1_MeanCandidate  = np.zeros((MaxFC, Method), dtype=int)

        r = 0

        for SC in range (NumSubCluster, NumSubCluster+1, 1):
            print('\n--> HProb_Raw+γstd <---')
            fn = './output_TwoStage_Search/Candidate_Lists/%d_%d.npy'%(FC, SC)
            fn2 = './output_candidate_lists_100k/%d_%d.npy'%(FC, SC)
            fn3 = './output_candidate_count/%d_%d.txt'%(FC, SC)

            InsIdx2 = createIndex(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, \
                                   p_prob, Zscore_var, testcases, FC, SC, model) 
            np.save(fn, InsIdx2)

            Top1Recall[r,0], Top1_MeanCandidate[r,0] = Searching_Top1(Ints_clustMem, Top1, idxCount, InsIdx2, FC, SC, fn2, fn3)      
          
            r +=1

        fN_Top1 = './output_TwoStage_Search/Recall/Recall_Top1_FC%d.txt'%(FC)
        np.savetxt(fN_Top1, Top1Recall, delimiter=' ', fmt='%1.4f')
        CTop1_Recall[cluster_idx]  = Top1Recall[0]

        CTop1_MeanCandidate[cluster_idx]  = Top1_MeanCandidate[0]

        cluster_idx += 1
    
    # Recall
    fN_CTop1 = './output_TwoStage_Search/Recall_Top1_%d_Clusters.txt'%(MaxFC)
    np.savetxt(fN_CTop1, CTop1_Recall, delimiter=' ', fmt='%1.4f')

    # MeanCandidate
    fN_CTop1_MeanCandidate = './output_TwoStage_Search/MeanCandidate_%d_Clusters.txt'%(MaxFC)
    np.savetxt(fN_CTop1_MeanCandidate, CTop1_MeanCandidate, delimiter=' ', fmt='%d')



# Apply early termination
def SearchingEarlyTer(MaxFC, NumSubCluster, Zscore_var, Ints_clustMem, idxCount, Top1, FirstStage_Codebook, SecondStage_Codebook, START, STEP):
    print('\nSearching with early termination\n')   
    model = ResidualModel()
    model.load_weights(args.ResidualModel_Weight_Path)
    
    query_sift = Load_file(args.Test_Loc+args.Test_Data_Path)
    learning_idx = Load_file(args.Output_Loc+args.First_pIdx_Path).astype(int)
    p_prob = Load_file(args.Output_Loc+args.First_pProb_Path)

    p_ter = Load_file(args.Output_Loc+args.First_pTermination_Path)
    p_ter /= np.max(p_ter)
    p_ter = np.array(np.around(p_ter*MaxFC), dtype=np.int)
    for i in range(p_ter.shape[0]):
        if p_ter[i] == 0:
            p_ter[i] = 1

    Method = 1
    testcases = []
    
    CTop1_Recall = np.zeros((1, Method), dtype=float)
    CTop1_MeanCandidate = np.zeros((1, Method), dtype=float)
    
    print('\n--> HProb_Raw+γstd <---')
    fn = './output_TwoStage_Search/Candidate_Lists/ET_%d_%d.npy'%(MaxFC, NumSubCluster)
    fn2 = './output_candidate_lists_100k/ET_%d_%d.npy'%(MaxFC, NumSubCluster)
    fn3 = './output_candidate_count/ET_%d_%d.txt'%(MaxFC, NumSubCluster)

    InsIdx2 = createIndex_EarlyTermination(query_sift, FirstStage_Codebook, SecondStage_Codebook, learning_idx, \
                                            p_prob, Zscore_var, testcases, MaxFC, NumSubCluster, model, p_ter)
    np.save(fn, InsIdx2)
 
    CTop1_Recall[0,0], CTop1_MeanCandidate[0,0] = Searching_Top1(Ints_clustMem, Top1, idxCount, InsIdx2, MaxFC, NumSubCluster, fn2, fn3)      

    # Recall
    fN_CTop1 = './output_TwoStage_Search/ET_Recall_Top1_%d_Clusters.txt'%(MaxFC)
    np.savetxt(fN_CTop1, CTop1_Recall, delimiter=' ',fmt='%1.4f')

    # MeanCandidate
    fN_CTop1_MeanCandidate = './output_TwoStage_Search/ET_MeanCandidate_%d_Clusters.txt'%(MaxFC)
    np.savetxt(fN_CTop1_MeanCandidate, CTop1_MeanCandidate, delimiter=' ', fmt='%d')
