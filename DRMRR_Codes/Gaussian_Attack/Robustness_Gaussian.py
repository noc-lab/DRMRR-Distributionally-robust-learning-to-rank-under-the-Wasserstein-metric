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
import pandas as pd
from six.moves import range
import random
import copy
import matplotlib.pyplot as plt
import pickle
from Robustness_Functions import Evaluation_LamdaMART_MAP_NDCG,Evaluation_LamdaMART_XE
from LTR_Functions import Read_LTR_Dataset
from DRMRR_Functions import Evaluation_DRMRR


# --------------------------------------------------
#    @@@@@@@@@@@@@@  Parameters  @@@@@@@@@@@@@@
# --------------------------------------------------
Percent_Sample_Noise = 0.75   
SD  = 0.001 
MUs = [i/1000 for i in range(0,251,10)]
FOLDS = [1,2,3,4,5]


# --------------------------------------------------
#  @@@@@@@@@@@@@@@@@@@  Data  @@@@@@@@@@@@@@@@@@@@
# -------------------------------------------------- 
Dir_Address= '/Data_OHSUMED/QueryLevelNorm/Fold'+str(1) # OHSUMED data directory location 
# Load data train datsets =========---------------------------------
with open(Dir_Address+"/train.txt") as trainfile:
    TX, Ty, Tqids, Tc = Read_LTR_Dataset(trainfile, has_targets=True , one_indexed=True)

# Load data Validation datsets =========---------------------------- vali
with open(Dir_Address+"/vali.txt") as validationfile:
    VX, Vy, Vqids, Vc = Read_LTR_Dataset(validationfile, has_targets=True , one_indexed=True)

# Load data test datsets =========----------------------------------
with open(Dir_Address+"/test.txt") as testfile:
    TestX, Testy, Testqids, Testc = Read_LTR_Dataset(testfile, has_targets=True , one_indexed=True)

# All data
All_X = np.vstack([TX,VX,TestX])
All_Y = np.hstack([Ty,Vy,Testy])
All_QID = np.hstack([Tqids,Vqids,Testqids])
All_QID = All_QID.astype(np.float)    # COnvert QIDs to float
All_Y_QID_X = np.hstack( [np.expand_dims(All_Y, axis=1), np.expand_dims(All_QID, axis=1) ,All_X ])  # Y-QID-X

All_Unique_QID=np.unique(All_QID)  # Unique Queriy IDs
A1 = './DRMRR_Paper_Codes/DRMRR_Model' # Generated data directory location - See Main_DRMRR.py
with open(A1+'/Final_X_OHSUMED_New_Output.pkl','rb') as f1:
    Final_X = pickle.load(f1)
with open(A1+'/Final_Relevance_OHSUMED_New_Output.pkl','rb') as f2:
    Final_Relevance = pickle.load(f2)
with open(A1+'/Final_Y_OHSUMED_New_Output.pkl','rb') as f3:
    Final_Y = pickle.load(f3)        
  
# Delete some redundant Vars 
del All_QID, All_Unique_QID, All_X, All_Y, All_Y_QID_X
del validationfile, trainfile, testfile, TX, Ty, Tc, VX, Vy, Vc, TestX, Testy, Testc
del f1,f2,f3,A1,Dir_Address




# --------------------------------------------------
#    @@@@@@@@@@@@  Gaussian Noise  @@@@@@@@@@@@
# --------------------------------------------------
counter=1
Count_Ep = 1
           
for mu in MUs:        
    print(mu)
    for fold in FOLDS:
        print('**** FOLD', fold, '  -----------------------')
        #  Best result paprameters
        # =========-------------------------------------------------------------------  
        #  Load the trained models
        Add_Theta = './Trained_Models'  # Trained models directory location
        Best_Theta_DRO_MRR = np.load(Add_Theta+'/MED_DRMRR/Best_Theta_DRMRR_LOSS_L_InfNone_Fold_'+str(fold)+'.npy')   
        
        with open(Add_Theta+'/MED_LamdaMart_MAP/Best_Model_LamdaMart_Fold_'+str(fold)+'.pkl','rb') as f1:
            Best_MODEL_LamdaMart_MAP = pickle.load(f1)
        with open(Add_Theta+'/MED_LamdaMart_NDCG/Best_Model_LamdaMart_Fold_'+str(fold)+'.pkl','rb') as f2:
            Best_MODEL_LamdaMart_NDCG = pickle.load(f2)    
        with open(Add_Theta+'/MED_XE_NDCG_MART/Best_Model_LamdaMart_Fold_'+str(fold)+'.pkl','rb') as f3:
            Best_MODEL_XE_NDCG_MART = pickle.load(f3)    
        
        del f1,f2,f3,Add_Theta

        # Load data test datsets =========----------------------------------
        Dir_Address= '/Data_OHSUMED/QueryLevelNorm/Fold'+str(fold) # OHSUMED data directory location 
        with open(Dir_Address+"/test.txt") as testfile:
            TestX, Testy, Testqids, Testc = Read_LTR_Dataset(testfile, has_targets=True , one_indexed=True)
        Test_Unique_QID = np.unique(Testqids).astype(np.float64)    
        
        
        # Perturbing data - FGSM-Regression
        Perturbed_Final_X_SimpReg = copy.deepcopy(Final_X)

        random.seed(2022)
        Noisy_Test_Unique_QID = list(random.choices(Test_Unique_QID,k=round(len(Test_Unique_QID)*Percent_Sample_Noise)))    
        for i in range(len(Final_X)):
            if Final_X[i,0] in Noisy_Test_Unique_QID:
                Perturbed_Final_X_SimpReg[i,1:] = Final_X[i,1:] + np.random.normal(mu, SD, (len(Final_X[i,:])-1))       
        
        # DRO-MRR                
        Perturbed_DR_MRR_Result = Evaluation_DRMRR(Test_Unique_QID,Best_Theta_DRO_MRR,Perturbed_Final_X_SimpReg,Final_Relevance)
        Perturbed_All_Result = pd.DataFrame(np.expand_dims(Perturbed_DR_MRR_Result, axis=0), columns=['NDCG@1', 'NDCG@5', 'NDCG@10', 'NDCG@25', 'NDCG@50', 'NDCG@100',
                                                    'Mean_Reciprocal_Rank' ,
                                                    'MAP@1', 'MAP@5', 'MAP@10', 'MAP@25', 'MAP@50', 'MAP@100'], index =['DRMRR'])
    
        # LamdaMart_MAP                  
        Perturbed_LamdaMart_MAP_Result = Evaluation_LamdaMART_MAP_NDCG(Test_Unique_QID,Best_MODEL_LamdaMart_MAP,   np.hstack( [Final_Relevance, Perturbed_Final_X_SimpReg ])   )
        Perturbed_All_Result.loc['LamdaMart_MAP'] = list(Perturbed_LamdaMart_MAP_Result)
    
        # LamdaMart_NDCG                
        Perturbed_LamdaMart_NDCG_Result =  Evaluation_LamdaMART_MAP_NDCG(Test_Unique_QID,Best_MODEL_LamdaMart_NDCG,   np.hstack( [Final_Relevance, Perturbed_Final_X_SimpReg ])   )
        Perturbed_All_Result.loc['LamdaMart_NDCG'] = list(Perturbed_LamdaMart_NDCG_Result)
    
         # XE_NDCG_MART                 
        Perturbed_XE_NDCG_MART_Result = Evaluation_LamdaMART_XE(Test_Unique_QID,Best_MODEL_XE_NDCG_MART,    np.hstack( [Final_Relevance, Perturbed_Final_X_SimpReg ])   )
        Perturbed_All_Result.loc['XE_NDCG_MART'] = list(Perturbed_XE_NDCG_MART_Result)     
        
        if fold == 1:
            Folds_Average_Perturbed_All = (Perturbed_All_Result)
        else:
            Folds_Average_Perturbed_All += (Perturbed_All_Result)
        counter+=1
      
        del Perturbed_DR_MRR_Result,
        del TestX, Testy, Testc,Perturbed_All_Result,
    
    A = ['Fold'+str(mu)]
    Fold_Row = pd.DataFrame({'NDCG@1':A, 'NDCG@5':A, 'NDCG@10':A,'NDCG@25':A, 'NDCG@50':A, 'NDCG@100':A,
                           'Mean_Reciprocal_Rank':A ,'MAP@1':A, 'MAP@5':A,'MAP@10':A, 'MAP@25':A, 'MAP@50':A, 'MAP@100':A}, index =[0])
    Folds_Average_Perturbed_All = Folds_Average_Perturbed_All/len(FOLDS)
    
    if Count_Ep == 1:
        Folds_Perturbed_All_Result = pd.concat([Fold_Row, Folds_Average_Perturbed_All]).reset_index(drop = True)
    else:
        Folds_Perturbed_All_Result = Folds_Perturbed_All_Result.append(pd.concat([Fold_Row, Folds_Average_Perturbed_All]).reset_index(drop = True) , ignore_index=True)

    Count_Ep += 1
    del i,Perturbed_XE_NDCG_MART_Result,Perturbed_LamdaMart_NDCG_Result,Perturbed_LamdaMart_MAP_Result
    del Dir_Address,A,Fold_Row,Folds_Average_Perturbed_All
    
del Tqids,Vqids,Testqids, testfile,Test_Unique_QID,counter,Count_Ep  
    


# --------------------------------------------------
#    @@@@@@@@@@@@@@   Plotting   @@@@@@@@@@@@@@
# --------------------------------------------------
Graph_Style = 'bmh'
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'smaller',
          #'figure.figsize': (15, 15),
         'axes.labelsize': 'x-small',
         'axes.titlesize':'smaller',
         'xtick.labelsize':'x-small',
         'ytick.labelsize':'x-small'}
pylab.rcParams.update(params)

with plt.style.context(Graph_Style):
    for N in [5,10]: 
        Col = 'NDCG@'+str(N)
        
        DRMRR_50 = [Folds_Perturbed_All_Result[Col][i] for i in range(1,len(Folds_Perturbed_All_Result),5)]
        LamdaMart_MAP_50 = [Folds_Perturbed_All_Result[Col][i] for i in range(2,len(Folds_Perturbed_All_Result),5)]
        LamdaMart_NDCG_50 = [Folds_Perturbed_All_Result[Col][i] for i in range(3,len(Folds_Perturbed_All_Result),5)]
        XE_NDCG_MART_50 = [Folds_Perturbed_All_Result[Col][i] for i in range(4,len(Folds_Perturbed_All_Result),5)]
        
        fig1, ax1 = plt.subplots(dpi=400)
        ax1.plot(MUs,DRMRR_50, color = 'forestgreen', label = 'DRMRR',linewidth=1.3)
            
        ax1.plot(MUs,LamdaMart_MAP_50, color = 'darkblue', label = 'LamdaMART-MAP',linewidth=1.3)
        ax1.plot(MUs,LamdaMart_NDCG_50, color = 'crimson', label = 'LamdaMART-NDCG',linewidth=1.3)
        ax1.plot(MUs,XE_NDCG_MART_50, color = 'darkturquoise', label = 'XE-MART-NDCG',linewidth=1.3)
        plt.legend(loc='lower left', prop={'size': 7})
        plt.title('Gaussian Noise Attack', fontweight="bold")
        plt.xlabel('Mean of Gaussian Noise',size=10)
        plt.ylabel(Col,size=10)
        y_min, y_max = ax1.get_ylim()
        ax1.set_ylim([(0.98*y_min),(1.01*y_max)])
        x_min, x_max = ax1.get_xlim()
        ax1.set_xlim([0,x_max])
        plt.show()


with plt.style.context(Graph_Style):   
    for N in  [5,10]: 
        Col = 'MAP@'+str(N)

        DRMRR_MAP_50 = [Folds_Perturbed_All_Result[Col][i] for i in range(1,len(Folds_Perturbed_All_Result),5)]
        LamdaMart_MAP_MAP_50 = [Folds_Perturbed_All_Result[Col][i] for i in range(2,len(Folds_Perturbed_All_Result),5)]
        LamdaMart_NDCG_MAP_50 = [Folds_Perturbed_All_Result[Col][i] for i in range(3,len(Folds_Perturbed_All_Result),5)]
        XE_NDCG_MART_MAP_50 = [Folds_Perturbed_All_Result[Col][i] for i in range(4,len(Folds_Perturbed_All_Result),5)]
        
        fig2, ax2 = plt.subplots(dpi=400)
        ax2.plot(MUs,DRMRR_MAP_50, color = 'forestgreen', label = 'DRMRR',linewidth=1.3)

        ax2.plot(MUs,LamdaMart_MAP_MAP_50, color = 'darkblue', label = 'LamdaMART-MAP',linewidth=1.3)
        ax2.plot(MUs,LamdaMart_NDCG_MAP_50, color = 'crimson', label = 'LamdaMART-NDCG',linewidth=1.3)
        ax2.plot(MUs,XE_NDCG_MART_MAP_50, color = 'darkturquoise', label = 'XE-MART-NDCG',linewidth=1.3)
        plt.legend(loc='lower left', prop={'size': 7})
        plt.title('Gaussian Noise Attack', fontweight="bold")
        plt.xlabel('Mean of Gaussian Noise',size=10)
        plt.ylabel(Col[1:],size=10)
        y_min, y_max = ax2.get_ylim()
        ax2.set_ylim([(0.98*y_min),(1.01*y_max)])
        x_min, x_max = ax2.get_xlim()
        ax2.set_xlim([0,x_max])
        plt.show()
    
    




