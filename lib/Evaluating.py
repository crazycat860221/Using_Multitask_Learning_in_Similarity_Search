import numpy as np

def Searching_Top1(Ints_clustMem, Top1, idxCount, InsIdx2, FC, SC, fn2, fn3):
    
    recall = np.zeros((Top1.shape[0]), dtype=int)
    len_lcl = np.zeros((Top1.shape[0]), dtype=int)
    Idx = np.zeros((len(InsIdx2), FC*SC), dtype=int)
    Idx -= 1

    print("\nComputing Top 1 recall:")
    for i in range(len(InsIdx2)): 
        for j in range(len(InsIdx2[i])):
            Idx[i,j] = InsIdx2[i,j]
            if idxCount[InsIdx2[i,j]] > 0 and InsIdx2[i,j] != -1: 
                ints_member = Ints_clustMem[InsIdx2[i,j]]
                lenMem = len(ints_member)
                len_lcl[i] += lenMem
                found = len(np.intersect1d(Top1[i], ints_member))
                recall[i] += found
                if len_lcl[i] > 100000: 
                    break
    
    TotalRecall =np.sum(recall) 
    AverageRecall = TotalRecall/Top1.shape[0]
    MeanCandidate = np.mean(len_lcl)
    
    print('================================')
    print('No. First Stage codebook : ', FC)
    print('No. Second Stage codebook : ', SC)
    print('Total Recall ', TotalRecall)
    print('AverageRecall ', AverageRecall)
    print('Mean Candidate ', int(MeanCandidate))
    print('================================')
    
    np.save(fn2, Idx)
    np.savetxt(fn3, len_lcl, delimiter=' ', fmt='%d')
    return AverageRecall, int(MeanCandidate)

