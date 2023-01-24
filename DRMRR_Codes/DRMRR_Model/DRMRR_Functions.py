"""
*************************************************************************
*************************************************************************
    Python implementation of DRMRR algorithm introduced in the paper: 
 "Distributionally robust learning-to-rank under the Wasserstein metric" 
     Shahabeddin Sotudian, Ruidi Chen, and Ioannis Ch. Paschalidis
*************************************************************************
*************************************************************************
"""


import numpy as np
import random
import copy
from scipy.stats import rankdata
from LTR_Functions import NDCG_K,MRR,AP_K
from LTR_Functions import Read_LTR_Dataset
import pickle
import pandas as pd


def DRMRR_Prediction(Test_X_features  , Theta_pp , Ground_Q):
    for k in range(len(Test_X_features)):
        P1 = np.matmul(np.transpose(Theta_pp) , Test_X_features[k,:]) 
       
        if k == 0:
            Pred = P1                       
        else:
            Pred = np.vstack([Pred, P1 ])  # each row: NDCG scores for Xi
                  
    Final_Rank=[]   
    for h1 in range(len(Test_X_features)):
        if h1%(np.shape(Pred)[1]) == 0:
            CC = 0
        else:
            CC += 1

        Final_Rank.append(np.argmax(Pred[:,CC], axis=0))
        Pred[np.argmax(Pred[:,CC], axis=0),:]= -np.ones([1,np.shape(Theta_pp)[1]])
    Concat = np.hstack([ Ground_Q  ,  np.expand_dims(Final_Rank, axis=1)  ])
    RGT =  np.array(  [Concat[int(Concat[m, 1]), 0] for m in range(len(Concat[:, 1]))]  )
       
    return RGT 

def Evaluation_DRMRR(Test_Unique_QID,BEST_Thetas,Final_X,Final_Relevance):
    Test_Num_Queries = len(Test_Unique_QID)
    Predicted_NDCGatk = np.array([1, 5, 10, 25, 50, 100, 111 ,1, 5, 10, 25, 50, 100])       
    NDCG_all = [0]*len(Predicted_NDCGatk)   
    for tt in range(Test_Num_Queries):
        Index_ID=np.where(Final_X[:,0] == Test_Unique_QID[tt])[0]  
        Test_X_features = Final_X[Index_ID,1:]
        Ground_Q= Final_Relevance[Index_ID,:]
        RGT= np.array( DRMRR_Prediction(Test_X_features , BEST_Thetas  , Ground_Q )   )  # Prediction in iteration pp-th
        # Performance Metrics
        Set_NDCG = [NDCG_K(RGT, i) for i in [1, 5, min(10,len(RGT)), min(25,len(RGT)), min(50,len(RGT)), min(100,len(RGT))]]  
        Set_Mean_Reciprocal_Rank = [MRR(RGT)]
        #print(len(RGT))
        Set_Average_Precision = [AP_K(RGT, i) for i in [1, 5, min(10,len(RGT)), min(25,len(RGT)), min(50,len(RGT)), min(100,len(RGT))]]
        All_Metrics = Set_NDCG + Set_Mean_Reciprocal_Rank + Set_Average_Precision
        NDCG_all = np.add(NDCG_all, All_Metrics)  
           
    Predicted_NDCG=NDCG_all/Test_Num_Queries
    return Predicted_NDCG

def F1_By_F2_Scores(Alpha,Beta,Max_r,DP,r):
    F1 = Alpha/(np.sqrt(abs(np.cosh(np.minimum((Beta)*DP,(Beta/2)*DP)))))
    F2 = np.log(Max_r*r+1)/np.log(Max_r*Max_r+1)
    return F1*F2


def Data_Prepration_GTD(Big_X_Relevance, OutPut_Length,Alpha,Beta,Max_r):
    AtK = OutPut_Length
    Index_Doc= np.expand_dims( np.linspace(0,(len(Big_X_Relevance)-1),len(Big_X_Relevance)) , axis=1) 
    Optimal_Rank= np.expand_dims( (rankdata(-Big_X_Relevance, method='ordinal')) , axis=1)    
    Concat= np.concatenate((Index_Doc,Optimal_Rank,Big_X_Relevance), axis=1)
    sorted_array = Concat[np.argsort(Concat[:, 1])]
    
    Optimal_Doc_Index = sorted_array[:,0]
    Optimal_Relevance = sorted_array[:,2]
    Multivariate_GTD= np.ones([len(Big_X_Relevance),OutPut_Length])
    for j1 in range(len(Big_X_Relevance)):
        Ideal_Position = np.where(Optimal_Doc_Index == j1)[0][0]
        for j2 in range(min(len(Optimal_Relevance),OutPut_Length)):
            input_seq =  copy.deepcopy(Optimal_Relevance)   
            input_seq[Ideal_Position], input_seq[j2] = input_seq[j2], input_seq[Ideal_Position]
            r = Optimal_Relevance[Ideal_Position]
            DP = Ideal_Position-j2
            Score1 = F1_By_F2_Scores(Alpha,Beta,Max_r,DP,r)
            Multivariate_GTD[j1 , j2] =  Score1*NDCG_K(input_seq, AtK) 
    return  Multivariate_GTD  # rows for diff docs - columns for diff positions 


def Reqularizer_DRMRR(THETA,OutPut_Length,Epsilon,L_lipschitz, Reg ='Reg_1_inf'):
    if Reg == 'Reg_1_inf':
        Reg_DRMRR = np.zeros(np.shape(THETA))
        Sum_Rows = np.sum(np.abs(THETA), axis=1)
        Max_Col = np.argmax(Sum_Rows, axis=None)
        if np.max(Sum_Rows)<1:
            return Reg_DRMRR
        else:
            Reg_DRMRR[Max_Col,:] = Epsilon *L_lipschitz*np.sign(THETA)[Max_Col,:]
            return Reg_DRMRR    

    elif Reg == 'Reg_inf_1':
        Reg_DRMRR = np.zeros(np.shape(THETA))
        Sum_Cols = np.sum(np.abs(THETA), axis=0)
        Max_Col = np.argmax(Sum_Cols, axis=None) 
        Reg_DRMRR[:,Max_Col] = Epsilon*L_lipschitz *np.sign(THETA)[:,Max_Col]
        return Reg_DRMRR 
            

def DRMRR_Gradient_Update(Final_X,Final_Y,Train_Unique_QID,THETA, OutPut_Length, Learning_Rate, Norm_p, Epsilon,L_lipschitz,Loss= 'L_Inf', Reg_Method ='OLD_L22'):
    if Loss == 'L_p':
        for i in range(len(Train_Unique_QID)):
            Index_ID=np.where(Final_X[:,0] == Train_Unique_QID[i])[0]  
            X_feat=Final_X[Index_ID,1:]
            X_Rel= Final_Y[Index_ID,:] 
            Total_Grad = np.zeros(np.shape(THETA))
            # Compute gradient    
            # A1: norm p 
            for j in range(len(X_feat)): # Compute gradient for i-th query - mini batch
                All_As = X_Rel[j,:] - np.matmul( np.transpose(THETA), X_feat[j,:] )
                Lp_Norm = np.linalg.norm(All_As, Norm_p)**(1-Norm_p)    
                Total_Grad = Total_Grad +  (-Lp_Norm * np.transpose([X_feat[j,:]] * len(All_As))) * (np.sign(All_As) * (All_As**(Norm_p-1)))
              
            Delta_Theta = Total_Grad*(1/len(X_Rel)) + Reqularizer_DRMRR(THETA,OutPut_Length,Epsilon,L_lipschitz, Reg =Reg_Method)
            THETA = THETA - (Learning_Rate * Delta_Theta)

    elif Loss == 'L_Inf':
        for i in range(len(Train_Unique_QID)):
            Index_ID=np.where(Final_X[:,0] == Train_Unique_QID[i])[0]  
            X_feat=Final_X[Index_ID,1:]
            X_Rel= Final_Y[Index_ID,:] 
            Total_Grad = np.zeros(np.shape(THETA))
            # Compute gradient
            for j in range(len(X_feat)): # Compute gradient for i-th query - mini batch
                All_As = X_Rel[j,:] - np.matmul( np.transpose(THETA), X_feat[j,:] )     
                Max_Columns = list(np.where(np.abs(All_As) == np.abs(All_As).max())[0])
                Max_Col = Max_Columns[random.randrange(len(Max_Columns))]
                Total_Grad[:,Max_Col]+=-X_feat[j,:]* np.sign(All_As)[Max_Col]
                
            Delta_Theta = Total_Grad*(1/len(X_Rel)) + Reqularizer_DRMRR(THETA,OutPut_Length,Epsilon,L_lipschitz, Reg =Reg_Method)
            THETA = THETA - (Learning_Rate * Delta_Theta)

    return THETA     


    
def DRMRR_5FCV_Func(Dir_Address,fold,Learning_rates_All,EPSILON_All,Num_itr,Norm_p,Type_Of_Loss,Type_Of_Regularizer,L_lipschitz,Alpha,Beta,OutPut_Length):     
    # Load data train datsets =========---------------------------------
    with open(Dir_Address+"/train.txt") as trainfile:
        TX, Ty, Tqids, Tc = Read_LTR_Dataset(trainfile, has_targets=True , one_indexed=True)
    
    # Load data Validation datsets =========---------------------------- vali
    with open(Dir_Address+"/vali.txt") as validationfile:
        VX, Vy, Vqids, Vc = Read_LTR_Dataset(validationfile, has_targets=True , one_indexed=True)
    
    # Load data test datsets =========----------------------------------
    with open(Dir_Address+"/test.txt") as testfile:
        TestX, Testy, Testqids, Testc = Read_LTR_Dataset(testfile, has_targets=True , one_indexed=True)
    
    if fold ==1:
        # All data
        All_X = np.vstack([TX,VX,TestX])
        All_Y = np.hstack([Ty,Vy,Testy])
        All_QID = np.hstack([Tqids,Vqids,Testqids])
        All_QID = All_QID.astype(np.float)    
        All_Unique_QID=np.unique(All_QID)  

        Unified_All_Y_QID_X = np.hstack( [np.expand_dims(All_Y, axis=1), np.expand_dims(All_QID, axis=1) ,All_X ])  # Y-QID-X
        Max_r = np.max(All_Y)
        
        for i in range(len(All_Unique_QID)):
            Index_ID=np.where(Unified_All_Y_QID_X[:,1] == All_Unique_QID[i])[0]  
            Big_X_features=Unified_All_Y_QID_X[Index_ID,1:]  # First column is IDs
            Big_X_Relevance= np.expand_dims(Unified_All_Y_QID_X[Index_ID,0], axis=1)
        
            # Convert relevance score to vector of NDCGs
            Multivariate_GTD = Data_Prepration_GTD(Big_X_Relevance, OutPut_Length,Alpha,Beta,Max_r)  
            if i == 0:
                Final_X = Big_X_features
                Final_Y = Multivariate_GTD
                Final_Relevance = Big_X_Relevance
            else:
                Final_X = np.vstack([Final_X,Big_X_features])
                Final_Y = np.vstack([Final_Y,Multivariate_GTD])
                Final_Relevance = np.vstack([Final_Relevance,Big_X_Relevance])  
        
        del i,Big_X_features,Big_X_Relevance,Index_ID,Multivariate_GTD
        del Unified_All_Y_QID_X, All_QID, All_Unique_QID, All_X, All_Y
        del validationfile, trainfile, testfile, TX, Ty, Tc, VX, Vy, Vc, TestX, Testy, Testc

        with open('Final_X_OHSUMED.pkl','wb') as f:
            pickle.dump(Final_X, f)
        with open('Final_Y_OHSUMED.pkl','wb') as g:
            pickle.dump(Final_Y, g)
        with open('Final_Relevance_OHSUMED.pkl','wb') as h:
            pickle.dump(Final_Relevance, h) 
    else:
        with open('Final_X_OHSUMED.pkl','rb') as f1:
            Final_X = pickle.load(f1)
        with open('Final_Relevance_OHSUMED.pkl','rb') as f2:
            Final_Relevance = pickle.load(f2)
        with open('Final_Y_OHSUMED.pkl','rb') as f3:
            Final_Y = pickle.load(f3)        
            
        del f1,f2,f3    
 

 
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @@@@                     Main DRMRR Algorithm 
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   
    
    #  Training =====--------------- 
    All_Thetas = []
    Last_Thetas = []
    Counter = 0
    All_parameters = []
    for LR in Learning_rates_All:
        for Ep in EPSILON_All:    
            Train_Unique_QID = np.unique(Tqids).astype(np.float64) 
            Num_Features= Final_X.shape[1] - 1
            # Initialize Theta  
            Theta_t_1 = np.zeros([(Num_itr+1),Num_Features,OutPut_Length])
            Theta_t_1[0,:,:]=0.00001 * np.ones([Num_Features,OutPut_Length])

            for t in range(Num_itr):    
                print('OO Iteration ', t+1, '>>>>>>--------  ')                                        
                Theta_t_1[(t+1),:,:]= DRMRR_Gradient_Update(Final_X,Final_Y,Train_Unique_QID, Theta_t_1[t,:], OutPut_Length, LR, Norm_p,Ep,L_lipschitz,Loss = Type_Of_Loss, Reg_Method =Type_Of_Regularizer)
                # Shuffle queries
                random.seed(t)
                random.shuffle(Train_Unique_QID)  
               
            del t
            if Counter == 0:
                All_Thetas = Theta_t_1
                Last_Thetas = Theta_t_1[(Num_itr),:,:].reshape(1,Theta_t_1[(Num_itr),:,:].shape[0],Theta_t_1[(Num_itr),:,:].shape[1])
            else:
                All_Thetas = np.append(All_Thetas,Theta_t_1, axis=0)
                Last_Thetas = np.append(Last_Thetas,Theta_t_1[(Num_itr),:,:].reshape(1,Theta_t_1[(Num_itr),:,:].shape[0],Theta_t_1[(Num_itr),:,:].shape[1]), axis=0)
    
            All_parameters.append([Counter, LR, Ep])
            Counter += 1
            
    del Counter, Theta_t_1  
    
    #  Validation =====---------------    
    Validation_Unique_QID = np.unique(Vqids).astype(np.float64)    
    for pp in range(Last_Thetas.shape[0]):           
        Predicted_NDCG = Evaluation_DRMRR(Validation_Unique_QID,Last_Thetas[pp],Final_X,Final_Relevance)
        if pp==0:
            arr = Predicted_NDCG
        else:
            arr = np.vstack((arr,Predicted_NDCG))
               
    Predicted_NDCGatk_Valid = pd.DataFrame(arr, columns=['NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@25', 'NDCG@50', 'NDCG@100','Mean_Reciprocal_Rank' ,'MAP@1', 'MAP@5', 'MAP@10', 'MAP@25', 'MAP@50', 'MAP@100'])
    del pp , Predicted_NDCG,arr
              
    # Best result parameters
    Top5_Top10 = Predicted_NDCGatk_Valid['NDCG@5'] + Predicted_NDCGatk_Valid['NDCG@10']   # highest Top5 and Top10 NDCG performance
    Best_Itr = (np.where( Top5_Top10 == np.max(Top5_Top10))[0][0])  
    
    
    #  Testing =====---------------       
    Test_Unique_QID = np.unique(Testqids).astype(np.float64)  
    Predicted_NDCG_t = Evaluation_DRMRR(Test_Unique_QID,Last_Thetas[Best_Itr],Final_X,Final_Relevance)    
    Predicted_NDCGatk_Test = pd.DataFrame(np.expand_dims(Predicted_NDCG_t, axis=0), columns=['NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@25', 'NDCG@50', 'NDCG@100','Mean_Reciprocal_Rank' ,'MAP@1', 'MAP@5', 'MAP@10', 'MAP@25', 'MAP@50', 'MAP@100'])    
    print('\n ******     Test Performance     ******  \n',Predicted_NDCGatk_Test.T)
    print('\n ******     Best Parameters     ******  \n')
    print('Best Learning Rate: ',All_parameters[Best_Itr][1])
    print('Best EPSILON:',All_parameters[Best_Itr][2])
    
    del Predicted_NDCG_t
    
    #  Save Best Theta  
    Best_Theta = Last_Thetas[Best_Itr] 
    Name_save = 'Best_Theta_DRMRR_LOSS_'+Type_Of_Loss+str(Norm_p)+'_Fold_'+str(fold)+'.npy'
    np.save(Name_save, Best_Theta) 
    
    return Predicted_NDCGatk_Valid,Predicted_NDCGatk_Test


    

