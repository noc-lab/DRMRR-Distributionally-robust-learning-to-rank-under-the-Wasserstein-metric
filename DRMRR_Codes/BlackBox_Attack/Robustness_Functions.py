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
from scipy.stats import rankdata
from LTR_Functions import NDCG_K,MRR,AP_K
import pandas as pd
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense  
from keras.layers import Dropout
import tensorflow as tf

def Evaluation_LamdaMART_MAP_NDCG(Validation_Unique_QID,Learned_Model_i,Unified_All_Y_QID_X):
   Validation_Num_Queries = len(Validation_Unique_QID)
   Predicted_NDCGatk = np.array([1, 5, 10, 25, 50, 100, 333333 ,1, 5, 10, 25, 50, 100])       
   NDCG_all = [0]*len(Predicted_NDCGatk)   
   for tt in range(Validation_Num_Queries):
       Index_ID=np.where(Unified_All_Y_QID_X[:,1] == Validation_Unique_QID[tt])[0]
       dValid_CoMPlex = xgb.DMatrix( Unified_All_Y_QID_X[Index_ID,2:] , label= np.expand_dims(Unified_All_Y_QID_X[Index_ID,0], axis=1)   )  
       Pred_Q = pd.DataFrame(Learned_Model_i.predict(dValid_CoMPlex))
       Ground_Q= Unified_All_Y_QID_X[Index_ID,0]
       Pred_Ranks=(rankdata(-Pred_Q, method='ordinal'))    # Decreasing order
       Concat = np.hstack([ np.expand_dims(Ground_Q, axis=1)  ,  np.expand_dims(Pred_Ranks, axis=1)  ])
       sorted_array = Concat[np.argsort(Concat[:, 1])]
       RGT= sorted_array[:,0]
       # Performance Metrics
       Set_NDCG = [NDCG_K(RGT, i) for i in [1, 5, min(10,len(RGT)), min(25,len(RGT)), min(50,len(RGT)), min(100,len(RGT))]]
       Set_Mean_Reciprocal_Rank = [MRR(RGT)]
       Set_Average_Precision = [AP_K(RGT, i) for i in [1, 5, min(10,len(RGT)), min(25,len(RGT)), min(50,len(RGT)), min(100,len(RGT))]]
       All_Metrics = Set_NDCG + Set_Mean_Reciprocal_Rank + Set_Average_Precision
       NDCG_all = np.add(NDCG_all, All_Metrics)  
          
   Predicted_NDCG=NDCG_all/Validation_Num_Queries
   return Predicted_NDCG 

def Evaluation_LamdaMART_XE(Validation_Unique_QID,Learned_Model_i,Unified_All_Y_QID_X):
   Validation_Num_Queries = len(Validation_Unique_QID)
   Predicted_NDCGatk = np.array([1, 5, 10, 25, 50, 100, 333333 ,1, 5, 10, 25, 50, 100])       
   NDCG_all = [0]*len(Predicted_NDCGatk)   
   for tt in range(Validation_Num_Queries):
       Index_ID=np.where(Unified_All_Y_QID_X[:,1] == Validation_Unique_QID[tt])[0]
       Pred_Q = pd.DataFrame(Learned_Model_i.predict(Unified_All_Y_QID_X[Index_ID,2:]))
       Ground_Q= Unified_All_Y_QID_X[Index_ID,0]
       Pred_Ranks=(rankdata(-Pred_Q, method='ordinal'))    # Decreasing order
       Concat = np.hstack([ np.expand_dims(Ground_Q, axis=1)  ,  np.expand_dims(Pred_Ranks, axis=1)  ])
       sorted_array = Concat[np.argsort(Concat[:, 1])]
       RGT= sorted_array[:,0]
       # Performance Metrics
       Set_NDCG = [NDCG_K(RGT, i) for i in [1, 5, min(10,len(RGT)), min(25,len(RGT)), min(50,len(RGT)), min(100,len(RGT))]]
       Set_Mean_Reciprocal_Rank = [MRR(RGT)]
       Set_Average_Precision = [AP_K(RGT, i) for i in [1, 5, min(10,len(RGT)), min(25,len(RGT)), min(50,len(RGT)), min(100,len(RGT))]]
       All_Metrics = Set_NDCG + Set_Mean_Reciprocal_Rank + Set_Average_Precision
       NDCG_all = np.add(NDCG_all, All_Metrics)  
          
   Predicted_NDCG=NDCG_all/Validation_Num_Queries
   return Predicted_NDCG 


def FGSM_Func_SimpReg(THETA,Single_Doc_feat,Single_Doc_Rel,EPSILON_FGSM): 
    A1 = 2*np.matmul( THETA, np.transpose(THETA))*Single_Doc_feat
    A2 = -2*Single_Doc_Rel*THETA
    A3= A1+A2
    FGSM_Delta = EPSILON_FGSM * np.sign(A3)      
    return FGSM_Delta


def get_NN_model(n_inputs, n_outputs,params):
    model = Sequential()
    model.add(Dense(128, input_dim=n_inputs, kernel_initializer='he_uniform', activation=params['activation']))
    model.add(Dropout(0.2)) 
    model.add(Dense(128, activation=params['activation']))
    model.add(Dropout(0.2)) 
    model.add(Dense(64, input_dim=n_inputs, kernel_initializer='he_uniform', activation=params['activation']))
    model.add(Dropout(0.2)) 
    model.add(Dense(32, activation=params['activation']))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
    model.compile(loss=params['losses'], optimizer='Adam')
    return model  

def Create_NN_BlackBox_Model(Train_Unique_QID,BEST_Model,Final_X,Final_Relevance,Model,params):
    if Model == 'MRR_or_DRMRR':
        C=0
        for tt in range(len(Train_Unique_QID)):
            Index_ID=np.where(Final_X[:,0] == Train_Unique_QID[tt])[0]  
            Test_X_features = Final_X[Index_ID,1:]
            if tt == 0:
                X_SM = Test_X_features
            else:
                X_SM = np.vstack([X_SM, Test_X_features ])
                
            for k in range(len(Test_X_features)):
                P1 = np.matmul(np.transpose(BEST_Model) , Test_X_features[k,:]) 
                if C == 0:
                    Y_SM = P1
                else:
                    Y_SM = np.vstack([Y_SM, P1 ])  # each row: NDCG scores for Xi
                C += 1

    elif Model == 'LamdaMART_MAP_NDCG':
        for tt in range(len(Train_Unique_QID)):
            Index_ID=np.where(Final_X[:,0] == Train_Unique_QID[tt])[0] 
            dValid_CoMPlex = xgb.DMatrix( Final_X[Index_ID,1:] , label= np.expand_dims(Final_Relevance[Index_ID,:], axis=1)   )  
            dValid_CoMPlex.set_group(group=np.array([len(Index_ID)]).flatten()) # Grouping
            Pred_Q = pd.DataFrame(BEST_Model.predict(dValid_CoMPlex))
            if tt == 0:
                X_SM = Final_X[Index_ID,1:]
                Y_SM = Pred_Q
            else:
                X_SM = np.vstack([X_SM, Final_X[Index_ID,1:] ])   
                Y_SM = np.vstack([Y_SM, Pred_Q ])

    elif Model == 'XE_NDCG_MART':
       for tt in range(len(Train_Unique_QID)):
            Index_ID=np.where(Final_X[:,0] == Train_Unique_QID[tt])[0]
            Pred_Q = pd.DataFrame(BEST_Model.predict(Final_X[Index_ID,1:]))       
            if tt == 0:
                X_SM = Final_X[Index_ID,1:]
                Y_SM = Pred_Q
            else:
                X_SM = np.vstack([X_SM, Final_X[Index_ID,1:] ])   
                Y_SM = np.vstack([Y_SM, Pred_Q ])

    # Training NN Model

    NN_Model = get_NN_model(X_SM.shape[1], Y_SM.shape[1], params)
    NN_Model.fit(X_SM, Y_SM,batch_size= params['batch_size'] ,epochs=params['epochs'],verbose=0)
    return NN_Model





def Generate_adversary_NN_Samples(model, image,label , eps):
    image = np.expand_dims(image, axis=0)  
    # cast the image
    image = tf.cast(image, tf.float32)
	# record our gradients
    with tf.GradientTape() as tape:
        tape.watch(image)
        pred = model(image)
        loss = tf.keras.losses.MSE(label,pred)
    gradient = tape.gradient(loss, image)
    signedGrad = tf.sign(gradient)
    adversary = (image + (signedGrad * eps)).numpy()
    return adversary 
