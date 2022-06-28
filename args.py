class parser(object):
    def __init__(self):
        # PATH
        self.Input_Loc = './input/'
        self.Train_Loc = './input/Train_Data/'
        self.Test_Loc = './input/Test_Data/'
        self.Output_Loc = './output/'

        self.Train_Data_Path = 'Train.npy'
        self.Residual_Train_Path = 'Train_Residual_CR.npy'

        #= Codebooks
        self.First_Codebook_Path = 'FirstStageCodebook.npy'
        self.Second_Codebook_Path = 'SecondStageCodebook.npy'

        #= Model
        self.Model_Weight_Path = './model/Index/Model_weight'
        self.ResidualModel_Weight_Path = './model/Residual/Model_weight'

        #= Test & Output
        self.Test_Data_Path = 'Test.npy'
        self.First_pIdx_Path = 'p_idx.npy'
        self.First_pProb_Path = 'p_prob.npy'
        self.First_pTermination_Path = 'p_ter.npy'