#For ICDM review only, please do not distribute
#ALL RIGHTS RESERVED
import numpy as np
import matplotlib.pyplot as plt
from solverADMMmodel7 import myADMMSolver as ADMM7
from solverADMMmodel6 import myADMMSolver as ADMM6
from helper import computeF1Score,extractSlidingWindow,computeF1ScoreWithoutBestMatch,reassignY, upperToFull
import copy


# In[32]:

def solve(mts_X,mts_hatX,modelId=7,K=4,beta=2,gamma=10, lamb=1000,rho=1,w=5,n=5,maxIters=30, eps_abs = 1e-2,eps_rel = 1e-2,debug = False,evaluate = False,groundTruth_Y={},groundTruth_hatY={},groundTruth_Z={}):
# In[]:         Initialization

    X = extractSlidingWindow(mts_X,w=w,n=n)
    hatX = extractSlidingWindow(mts_hatX,w=w,n=n)
    returnVal = {}
    vec_thetas7 = {}
    vec_thetas6 = {}
    LOG2PI = np.log(2*np.pi)
    nw = n*w
    T = X.shape[0]
    hatT = hatX.shape[0]
    Y = np.zeros(T)
    hatY = np.zeros(hatT)
    Z = np.zeros(hatT)
    if debug:
        print "completed getting the data"
        print "lamb", lamb
        print "beta", beta
        print "gamma", gamma
        print "K", K
        print "w", w
        print "n",n
    theta = np.zeros([K,nw,nw])
    hatTheta = np.zeros([K,nw,nw])
    list_f1Y = []
    list_f1hatY = []
    list_f1Z = []
# In[] ############## Initialize latent state assignment ###############

    breakPoint = T/K
    if debug:
        print "breakPoint",breakPoint
    for k in xrange(K):
        if k<K-1:
            Y[breakPoint*k:breakPoint*(k+1)] = k
        else:
            Y[breakPoint*k:] = k




    if evaluate:
        print "initial Y assignment"
        random_f1_Y = computeF1Score(Y,groundTruth_Y, plotCluster=False)
        print "Initial_Y_f1 =", random_f1_Y
        list_f1Y.append(random_f1_Y)
    breakPoint = hatT/K
#    print "breakPoint",breakPoint
    for k in xrange(K):
        if k<K-1:
            hatY[breakPoint*k:breakPoint*(k+1)] = K-k-1
#                Z[int(breakPoint*(k+0.5)):breakPoint*(k+1)]=1
        else:
            hatY[breakPoint*k:] = K-k-1
    if evaluate:
        print "initial hatY assignment"
        random_f1_hatY = computeF1Score(hatY,groundTruth_hatY, plotCluster=True)
        print "Initial_hatY_f1 =", random_f1_Y
        print "initial Z assignment"
        initZ_f1 = computeF1ScoreWithoutBestMatch(Z,groundTruth_Z,plotCluster=True)
        print "Initial_Z_f1 =", initZ_f1
        list_f1hatY.append(random_f1_hatY)
        list_f1Z.append(initZ_f1)

    for iters in xrange(maxIters):     # start E-M algorithm
        print "\n E-M ITERATION ####################", iters
#       M-step:
#        old_Y = copy.deepcopy(Y)
        old_hatY = copy.deepcopy(hatY)
        old_Z = copy.deepcopy(Z)
        list_x = {}   #dict{cluster: list of subsequences belong to the cluster}
        list_hatX = {}
        backup_list_hatX = {}
        list_x3 = {}


        for k in xrange (K):
            list_x[k] = []
            list_hatX[k] = []
            backup_list_hatX[k] = []
            list_x3[k] = []
        for t in xrange(T):
            k = Y[t]
            list_x[k].append(X[t,:])
            list_x3[k].append(X[t,:])

        for t in xrange(hatT):
            k = hatY[t]
            z = Z[t]
            backup_list_hatX[k].append(hatX[t,:])
            if z==0:
                list_hatX[k].append(hatX[t,:])

            else:
#                    print "do nothing"
                list_x[k].append(hatX[t,:])

        x = {}
        hat_x = {}
        x3 = {}
        dict_mu = {}
        dict_hat_mu = {}
        dict_x3_mu = {}

        for k in xrange(K):
            if len(list_hatX[k])<nw:   # in case of too small data points to estimate the precision matrix
                list_hatX[k] = backup_list_hatX[k][:10*nw]
            x[k] = np.asarray(list_x[k])
            dict_mu[k] = np.mean(x[k],axis=0)
            hat_x[k] = np.asarray(list_hatX[k])
            dict_hat_mu[k] = np.mean(hat_x[k],axis=0)
            x3[k] = np.asarray(list_x3[k])
            dict_x3_mu[k] = np.mean(x3[k],axis=0)
            C1 = x[k].shape[0]
            C0 = hat_x[k].shape[0]

            if debug:
                print "C0,C1",C0, C1
                plt.scatter(x[k][:,0],x[k][:,1],color='C0',s=0.5)
                plt.xlim(-10,10)
                plt.ylim(-5,5)
                plt.show()
                plt.scatter(hat_x[k][:,0],hat_x[k][:,1],color='C1',s=0.5)
                plt.xlim(-10,10)
                plt.ylim(-5,5)
                plt.show()
            if(x[k].shape[0]>nw and hat_x[k].shape[0]>nw):
                if(modelId==7):
                    solver7 = ADMM7(lamb,nw,rho,x[k],hat_x[k])
                    vec_thetas7[k] = solver7(eps_abs,eps_rel,False)
                else:
                    solver6 = ADMM6(lamb,nw,rho,x[k],hat_x[k])
                    vec_thetas6[k] = solver6(200,eps_abs,eps_rel,False)
            else:
                print "x[k].shape[0],hat_x[k].shape[0]",x[k].shape[0],hat_x[k].shape[0]

# In[]: E-step
        if(modelId == 7): # Model uses partial correlation based transformation  function
            vec_thetas = vec_thetas7
#            print "Model uses partial correlation based transformation  function"
        else: # Model uses Covariance based transformation function.
            vec_thetas = vec_thetas6
#            print "Model uses Covariance based transformation function."
        logDETtheta1_dict = {} # cluster to log_det
        logDETtheta2_dict = {} # cluster to log_det
#  compute log det
        for cluster in xrange(K):
            val = vec_thetas[cluster]
            val1 = val[:,0]
            val2 = val[:,1]
            if debug:
                print '@@@@@@ Iter',iters
                print 'Cluster',cluster
            if debug and not np.any(val2):  # check all elements are zeros
                print "val2.all()==0: cluster = ",cluster
    #            val2 = val1
#            print "OPTIMIZATION for Cluster #", cluster,"DONE!!!"
            theta1 = upperToFull(val1, 0)
            theta2 = upperToFull(val2, 0)
            cov = np.linalg.inv(theta1)
            hatCov = np.linalg.inv(theta2)
            if debug:
                print '@@@@@@ Iter',iters
                print 'Cluster',cluster
                print 'cov\n',cov
#                plotMultiGaussian(cov)
                print 'hatCov\n',hatCov
#                plotMultiGaussian(hatCov)

            DETtheta1 = np.linalg.det(theta1)
            DETtheta2 = np.linalg.det(theta2)
            if DETtheta1==0:
    #            print theta1
                print "???????????????????????np.linalg.det(theta1) = ",DETtheta1
                DETtheta1 = 0e-4#np.abs(DETtheta1);


            if DETtheta2==0:
    #            print theta2
                print "????????????????????????np.linalg.det(theta2) = ",DETtheta2
                DETtheta2 = 0e-4#np.abs(DETtheta2);


            logDETtheta1 = np.log(DETtheta1)
            logDETtheta2 = np.log(DETtheta2)
            logDETtheta1_dict[cluster] = logDETtheta1
            logDETtheta2_dict[cluster] = logDETtheta2

            theta[cluster] = theta1
            hatTheta[cluster] = theta2

        Y_changed_count = 0

        lle_allNodes = np.zeros([T,K])
        for t in xrange(T):
            for k in xrange(K):
                x_mu = X[t,:] - dict_mu[k]
                lle_allNodes[t,k] = (np.dot(np.dot(np.transpose(x_mu),theta[k]),x_mu)-logDETtheta1_dict[k]+nw*LOG2PI)/2

        Y = reassignY(lle_allNodes,gamma)
        if evaluate:
            print "Y assignment at iteration",iters
            f1_Y = computeF1Score(Y,groundTruth_Y,True)
            print "Y_f1 =", f1_Y
            print ""
            list_f1Y.append(f1_Y)

        Z_changed_count = 0
        # Start dynamic programming algorithm to reassign hatY and Z
        ll0 = np.zeros([hatT,K])   # corresponding to J in the paper
        ll1 = np.zeros([hatT,K])   # corresponding to \hat J in the paper
        for t in xrange(hatT):
            for k in xrange(K):
                x1 = hatX[t,:] - dict_mu[k]
                x0 = hatX[t,:] - dict_hat_mu[k]
                theta1 = theta[k]
                theta2 = hatTheta[k]
                if t == 0:
                    cov = np.linalg.inv(theta1)
                    hatCov = np.linalg.inv(theta2)
                logDETtheta1 = logDETtheta1_dict[k]
                logDETtheta2 = logDETtheta2_dict[k]

                ll0[t,k] = ((np.dot(np.dot(np.transpose(x0),theta2),x0)-logDETtheta2+(nw)*LOG2PI)/2)
                ll1[t,k] = ((np.dot(np.dot(np.transpose(x1),theta1),x1)-logDETtheta1+(nw)*LOG2PI)/2)
#
        preCost = np.zeros([K,2])
        curCost = np.zeros([K,2])
        #prePath = [[]]*(K*2)       # encoding 10 11 20 21... with 0 1 2 3 4...
        #curPath = [[]]*(K*2)
        pathMat = np.zeros([hatT,K*2],dtype = 'int32')
        #
        cost = []
        minNode = 0
        for t in xrange(hatT):
        #    if t%1000 == 0:
        #        print t

        #topTen = 1000;
        #for t in xrange(topTen):
        #    curPath = [[]]*(K*2)
            pminIdx0 = np.argmin(preCost[:,0])
            preMin0 = preCost[pminIdx0,0]
            pminIdx1 = np.argmin(preCost[:,1])
            preMin1 = preCost[pminIdx1,1]
            for k in xrange(K):
        #        when Z = 0
                curCost0 = preCost[k,0]
                curCost1 = preCost[k,1]+beta
                preMinCost0 = preMin0+gamma
                preMinCost1 = preMin1+beta+gamma
                cost = [curCost0, curCost1, preMinCost0, preMinCost1]
                minNode = np.argmin(cost)
                curCost[k,0] = cost[minNode]+ll0[t,k]
                if minNode == 0:
                    pathMat[t,k*2] = k*2

                elif minNode == 1:
                    pathMat[t,k*2] = k*2+1

                elif minNode == 2:
                    pathMat[t,k*2] = pminIdx0*2

                else:
                    pathMat[t,k*2] = pminIdx1*2+1

                curCost0 = preCost[k,0]+beta
                curCost1 = preCost[k,1]
                preMinCost0 = preMin0+beta+gamma
                preMinCost1 = preMin1+gamma
                cost = [curCost0, curCost1, preMinCost0, preMinCost1]

                minNode = np.argmin(cost)
                curCost[k,1] = cost[minNode]+ll1[t,k]
                if minNode == 0:
                    pathMat[t,k*2+1] = k*2
                elif minNode == 1:
                    pathMat[t,k*2+1] = k*2+1
                elif minNode == 2:
                    pathMat[t,k*2+1] = pminIdx0*2
                else:
                    pathMat[t,k*2+1] = pminIdx1*2+1
            preCost = curCost

        finalMinIdx = np.argmin(curCost)

        curNode = finalMinIdx
        preNode = finalMinIdx

        for t in xrange(hatT-1,-1,-1):
            preNode = pathMat[t,curNode]
            hatY[t] = preNode/2
            Z[t] = preNode%2
            if not(np.equal(hatY[t],old_hatY[t])):
                Y_changed_count = Y_changed_count+1
            if not(np.equal(Z[t],old_Z[t])):
                Z_changed_count = Z_changed_count+1
            curNode = preNode


# In[normal]
        if evaluate:
            print "hat_Y assignments at iteration:",iters
            f1_hatY = computeF1Score(hatY,groundTruth_hatY,True)
            print "hat_Y_f1 = ",f1_hatY
            print "Z assignments at iteration:",iters
            f1_hatZ = computeF1ScoreWithoutBestMatch(Z,groundTruth_Z,True)
            accZ = 1.0*sum(Z==groundTruth_Z)/hatX.shape[0]
            print "Z_f1, accZ = ",f1_hatZ,accZ
            list_f1hatY.append(f1_hatY)
            list_f1Z.append(f1_hatZ)


# In[check convergence]

        if (Y_changed_count<30 and Z_changed_count<30 and iters>5) :
                print "All Done!!!!!!!!!!!!!!!!!!!! Early stationary at iteration", iters
                break;
            # end E-M loop
# In[Evaluate results]:
    if evaluate:
        xaxis = xrange(len(list_f1Y))
        plt.figure()
    #    f1plt = plt.plot()
        plt.plot(xaxis,list_f1Y,label = "f1Y")
        plt.plot(xaxis,list_f1hatY,label = "f1hatY")
        plt.plot(xaxis,list_f1Z,label = "f1Z")
        plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("f1 score")

    returnVal['Y'] = Y
    returnVal['hatY'] = hatY
    returnVal['Z'] = Z
    returnVal['theta'] = theta
    returnVal['hatTheta'] = hatTheta

    return returnVal



