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
from six.moves import range
import pickle
from LTR_Functions import Read_LTR_Dataset
from DRMRR_Functions import Data_Prepration_GTD

# --------------------------------------------------
#    @@@@@@@@@@@@@@  Parameters  @@@@@@@@@@@@@@
# --------------------------------------------------
Noise_Type = 'High' #     High  Low 
Alpha = 50
Beta = 0.5
OutPut_Length = 5 
Seed_Random=1  


# --------------------------------------------------
#  @@@@@@@@@@@@@@@@@@@  Data  @@@@@@@@@@@@@@@@@@@@
# --------------------------------------------------
A1 = './DRMRR_Paper_Codes/DRMRR_Model'# Generated data directory location - See Main_DRMRR.py
with open(A1+'/Final_X_OHSUMED_New_Output.pkl','rb') as f1:
    Final_X = pickle.load(f1)
with open(A1+'/Final_Relevance_OHSUMED_New_Output.pkl','rb') as f2:
    Final_Relevance = pickle.load(f2)
with open(A1+'/Final_Y_OHSUMED_New_Output.pkl','rb') as f3:
    Final_Y = pickle.load(f3)        
    
    
# --------------------------------------------------
#    @@@@@@@@@@@@  Label Poisoning  @@@@@@@@@@@@@
# --------------------------------------------------

for fold in range(1,6,1):
    print('***********************************************')
    print('**** FOLD', fold, '  -----------------------')
    print('***********************************************')
    
    # Load data train data of Fold
    Dir_Address= '/Data_OHSUMED/QueryLevelNorm/Fold'+str(fold) # OHSUMED data directory location 
    with open(Dir_Address+"/train.txt") as trainfile:
        TX, Ty, Tqids, Tc = Read_LTR_Dataset(trainfile, has_targets=True , one_indexed=True)
    Train_Unique_QID = np.unique(Tqids).astype(np.float64)   
    del TX, Ty,Tc,trainfile
     
    # Load data test datsets =========----------------------------------
    with open(Dir_Address+"/test.txt") as testfile:
        TestX, Testy, Testqids, Testc = Read_LTR_Dataset(testfile, has_targets=True , one_indexed=True)
    Test_Unique_QID = np.unique(Testqids).astype(np.float64)    # Unique Queriy IDs + COnvert QIDs to float   Testqids Tqids    
    del TestX, Testy, Testc, testfile        
    
    # Load data Valid datsets =========----------------------------------
    with open(Dir_Address+"/vali.txt") as testfile:
        ValidX, Validy, Vqids, Validc = Read_LTR_Dataset(testfile, has_targets=True , one_indexed=True)
    Test_Unique_QID = np.unique(Testqids).astype(np.float64)    # Unique Queriy IDs + COnvert QIDs to float   Testqids Tqids    
    del ValidX, Validy, Validc, testfile,Dir_Address       
    
    
    R_max = int(np.max(Final_Relevance))
    if Noise_Type == 'Low':
        Prob_Rel = np.array([[0.85,0.1,0.05],[0.075,0.85,0.075],[0.05,0.1,0.85]])
    elif Noise_Type == 'High':
        Prob_Rel = np.array([[0.7,0.2,0.1],[0.15,0.7,0.15],[0.1,0.2,0.7]])
         

    # Add noise
    All_QID = np.hstack([Tqids,Vqids,Testqids])
    All_QID = All_QID.astype(np.float64)    # COnvert QIDs to float
    Unified_All_Y_QID_X = np.hstack( [Final_Relevance,Final_X ])  # Y-QID-X
    
    All_Unique_QID=np.unique(All_QID).astype(np.float64)  # Unique Queriy IDs
    Training_Unique_QID=np.unique(Tqids).astype(np.float64)
    
    AtK = len(Unified_All_Y_QID_X[np.where(Unified_All_Y_QID_X[:,1] == All_Unique_QID[0])[0],0])
    COUNT = 0
    for i in range(len(All_Unique_QID)):
        Index_ID=np.where(Unified_All_Y_QID_X[:,1] == All_Unique_QID[i])[0]  
        Big_X_features=Unified_All_Y_QID_X[Index_ID,1:]  # First column is IDs
        Big_X_Relevance= np.expand_dims(Unified_All_Y_QID_X[Index_ID,0], axis=1)
        # Add noise to label if training QID ----------------------------------
        if All_Unique_QID[i] in Training_Unique_QID:
            COUNT += 1
            print('add noise training sample: ', COUNT)
            for n in range(len(Big_X_Relevance)):
                
                np.random.seed(Seed_Random)
                Big_X_Relevance[n] = np.random.choice((R_max+1), 1, p=Prob_Rel[int(Big_X_Relevance[n]),:], replace = True)
                Seed_Random += 1
    
        # Convert relevance score to vector of NDCGs ----------------------------------
        Multivariate_Y = Data_Prepration_GTD(Big_X_Relevance, OutPut_Length,Alpha,Beta,R_max)
        
        if i == 0:
            Final_X_NOISE_Label = Big_X_features
            Final_Y_NOISE_Label = Multivariate_Y
            Final_Relevance_NOISE_Label = Big_X_Relevance
        else:
            Final_X_NOISE_Label = np.vstack([Final_X_NOISE_Label,Big_X_features])
            Final_Y_NOISE_Label = np.vstack([Final_Y_NOISE_Label,Multivariate_Y])
            Final_Relevance_NOISE_Label = np.vstack([Final_Relevance_NOISE_Label,Big_X_Relevance])  
    
    del i,Big_X_features,Big_X_Relevance,Index_ID,Multivariate_Y
    
    ###################### Delete some redundant Vars ####################################
    del All_QID, All_Unique_QID,Training_Unique_QID,n
    
    # save data
    with open(Noise_Type +'_'+'Final_X_NOISE_Label_Fold_'+str(fold)+'.pkl','wb') as f1:
        pickle.dump(Final_X_NOISE_Label, f1)
    with open(Noise_Type +'_'+'Final_Y_NOISE_Label_Fold_'+str(fold)+'.pkl','wb') as f2:
        pickle.dump(Final_Y_NOISE_Label, f2)
    with open(Noise_Type +'_'+'Final_Relevance_NOISE_Label_Fold_'+str(fold)+'.pkl','wb') as f3:
        pickle.dump(Final_Relevance_NOISE_Label, f3)
    
    with open(Noise_Type +'_'+'Final_Tqids_Fold_'+str(fold)+'.pkl','wb') as f4:
        pickle.dump(Tqids, f4)
    with open(Noise_Type +'_'+'Final_Vqids_Fold_'+str(fold)+'.pkl','wb') as f5:
        pickle.dump(Vqids, f5)
    with open(Noise_Type +'_'+'Final_Testqids_Fold_'+str(fold)+'.pkl','wb') as f6:
        pickle.dump(Testqids, f6)


