#For ICDM review only, please do not distribute
#ALL RIGHTS RESERVED
import numpy as np
from helper import computeF1Score,computeF1ScoreWithoutBestMatch
from solverFramework import solve


# In[parameters]:
[ADMMstop_threshold,w,n,lamb,beta,gamma,maxIter,rho] = [1e-2,5,5,1000,2,10,10,1]  # refer to our paper

# In[load data]:
folder = 'example_dataset/'
fname = folder+"X.txt"
oriDataset = np.loadtxt(fname, delimiter= ",")
mts_X = oriDataset[:,2:]
groundTruthY = oriDataset[:oriDataset.shape[0]-w+1,0]
hat_fname = folder+"hatX.txt"
hat_oriDataset = np.loadtxt(hat_fname, delimiter= ",")
mts_hatX = hat_oriDataset[:,2:]
groundTruthHatY = hat_oriDataset[:hat_oriDataset.shape[0]-w+1,0]
groundTruthZ = hat_oriDataset[:hat_oriDataset.shape[0]-w+1,1]
groundTruthY_dict = {0:groundTruthY}
groundTruthHatY_dict = {0:groundTruthHatY}
groundTruthZ_dict = {0:groundTruthZ}

# In[run the model]:

#CPM-P (modelId=7): Model with parital correlation based translation function:
predictedResult = solve(mts_X,mts_hatX,modelId=7,K=4,beta=beta,gamma=gamma,lamb=lamb,rho=rho,w=w,n=n,maxIters = maxIter,eps_abs = ADMMstop_threshold,eps_rel = ADMMstop_threshold,debug = False,evaluate=False)

#CPM-C (modelId= 6 ): Model with covariance based translation function
#predictedResult = solve(mts_X,mts_hatX,modelId=6,K=4,beta=beta,gamma=gamma,lamb=lamb,rho=rho,w=w,n=n,maxIters = maxIter,eps_abs = ADMMstop_threshold,eps_rel = ADMMstop_threshold,debug = False,evaluate=False)

# In[evaluate the result]:
predict_Y = predictedResult['Y']
predict_hatY = predictedResult['hatY']
predict_Z=predictedResult['Z']
print 'performance'
print 'Y assignment: macro F1 = ', computeF1Score(predict_Y,groundTruthY)
print 'hatY assignment: macro F1 = ',computeF1Score(predict_hatY,groundTruthHatY)
print 'Z assignment: F1 = ', computeF1ScoreWithoutBestMatch(predict_Z,groundTruthZ)
