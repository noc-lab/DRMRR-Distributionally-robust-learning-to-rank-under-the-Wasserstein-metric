
"""
*************************************************************************
*************************************************************************
    Python implementation of DRMRR algorithm introduced in the paper: 
 "Distributionally robust learning-to-rank under the Wasserstein metric" 
     Shahabeddin Sotudian, Ruidi Chen, and Ioannis Ch. Paschalidis
*************************************************************************
*************************************************************************

===================================================================================================================================================
Usage =========----------------------

Inputs:
   Alpha:                 GTD’s maximum score
   Beta:                  Regulates the magnitude of penalty for a position deviation
   K:                     Length of GTD vector
   Num_itr:               Number of iterations
   Learning_rates_Set:    Set of all possible learning rates
   EPSILON_Set:           Set of all possible epsilons
   Type_Of_Loss:          Type of loss function. Use 'L_Inf' for L-infinity norm or 'L_p' for L-p norm.
   Norm_p:                Degree of the norm (e.g., 1 for L-1 norm and 'None' for L-infinity norm)
   L_lipschitz:           Lipschitz constant of loss function
   Type_Of_Regularizer:   DRMRR regularizer that should be selected based on the type of loss function. Options: 'Reg_1_inf' or 'Reg_inf_1'

===================================================================================================================================================
"""
from DRMRR_Functions import DRMRR_5FCV_Func


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@                         5 FCV  - DRMRR
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

#  Parameters
# =========-------------------------------------------------------------------       
Alpha = 50
Beta = 0.5
K = 5   
Num_itr=50
Learning_rates_Set = [1E-1,1E-2,1E-3]
EPSILON_Set = [1,1E-1,1E-2,1E-3]
Data_Dir= '/Data_OHSUMED/QueryLevelNorm' # OHSUMED data directory location 


Type_Of_Loss = 'L_Inf' 
Norm_p = 'None'  
L_lipschitz = 1
Type_Of_Regularizer = 'Reg_1_inf'    


Performnace_Validation = {}
Performnace_Test = {}
for fold in range(1,6):
    print('\n\n************************************')
    print('**** FOLD', fold, '  -----------')
    print('************************************')
     
    Dir_Address= Data_Dir+'/Fold'+str(fold)                                                               
    Performnace_Validation["FOLD-{0}".format(fold)],Performnace_Test["FOLD-{0}".format(fold)] = DRMRR_5FCV_Func(Dir_Address,fold,Learning_rates_Set,EPSILON_Set,Num_itr,Norm_p,Type_Of_Loss,Type_Of_Regularizer,L_lipschitz,Alpha,Beta,K)

Performnace_Test['Average'] = sum(Performnace_Test.values()) / len(Performnace_Test)
del Alpha,Beta,K,Num_itr,Learning_rates_Set,EPSILON_Set,fold,Dir_Address,Type_Of_Loss,Norm_p,L_lipschitz,Type_Of_Regularizer



