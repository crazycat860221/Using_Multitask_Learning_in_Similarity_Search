import numpy as np
import tensorflow as tf

from lib.Load import Load_file
from lib.Network import IndexModel
from args import parser

args = parser()

def First_Predict():
    # Load query
    query = Load_file(args.Test_Loc+args.Test_Data_Path)
    query /= np.max(query)

    # Load model
    model = IndexModel()
    model.load_weights(args.Model_Weight_Path)

    # predict
    print("\n====== Testing Part ======")
    cate_pred, dist_pred = model(query)

    p_idx = np.argsort(-cate_pred)
    
    # Save p_prob, p_idx & clus_cand_num
    pProb_path = args.Output_Loc + args.First_pProb_Path
    np.save(pProb_path, cate_pred)

    pIdx_path = args.Output_Loc + args.First_pIdx_Path
    np.save(pIdx_path, p_idx)

    pTer_path = args.Output_Loc + args.First_pTermination_Path
    np.save(pTer_path, dist_pred)