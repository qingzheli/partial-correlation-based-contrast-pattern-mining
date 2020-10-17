#For ICDM review only, please do not distribute
#ALL RIGHTS RESERVED
#ADMM solver for CPM-C model with covariance based translation function
import numpy as np
import copy
import time
# In[]
class myADMMSolver:
    def __init__(self,lamb, nw, rho,x,hat_x,rho_update_func=None):
        self.nw = nw
        self.lamb = lamb
        self.rho = rho
        self.length = self.nw*(self.nw+1)/2  #vector length of trianglular matrix
        self.m = x.shape[0]
        self.hatm = hat_x.shape[0]
        self.S = np.cov(np.transpose(x))
        self.hatS = np.cov(np.transpose(hat_x))
        self.theta = np.eye(self.nw)  # self.length*1
        self.hat_theta = np.eye(self.nw)
        self.Q= np.eye(self.nw)
        self.hatQ = np.eye(self.nw)
        self.T = np.eye(self.nw)
        self.hatT = np.eye(self.nw)
        self.U1 = np.eye(self.nw)
        self.U2 = np.eye(self.nw)
        self.U3 = np.eye(self.nw)
        self.U4 = np.eye(self.nw)
        self.V = np.zeros((nw,nw))
        self.U5 = np.eye(self.nw)
        self.plist = []
        self.dlist = []
        self.eprilist = []
        self.edualist = []
        self.objlist = []
        self.rsmin = 10000000
        self.minObj = 10000000
        self.minTheta =  np.eye(self.nw)
        self.minHatTheta = np.eye(self.nw)


    def h(self,theta,S,m):
        return 0.5*m*(np.trace(np.dot(S,theta))-np.log(np.linalg.det(theta)))
    def obj_overall(self):
        obj = self.h(self.T,self.S,self.m)+self.h(self.hatT,self.hatS,self.hatm)+self.lamb*(np.linalg.norm(self.V)**2)
        if(obj<-100000):
            return np.inf
        else:
            return obj

    def ij2symmetric(self, i,j,size):
        return (size * (size + 1))/2 - (size-i)*((size - i + 1))/2 + j - i

    def upper2Full(self, vecA):  #  vector a  to symetric matrix A
        a = np.reshape(vecA,[vecA.shape[0]])
        n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)
        A = np.zeros([n,n])
        A[np.triu_indices(n)] = a
        temp = A.diagonal()
        A = (A + A.T) - np.diag(temp)
        return A


    def prox_positive_definite(self,A):
        LAMBDA,D = np.linalg.eigh(A)
        retVal = np.zeros(A.shape)
        n  = A.shape[0]
        for i in xrange(n):
            if LAMBDA[i]<0:
                LAMBDA[i] = 0
            retVal = retVal+ LAMBDA[i]*np.dot(D[:,i],np.transpose(D[:,i]))
        return retVal
    def proj2Symmetric(self,A):
        n  = A.shape[0]
        for i in xrange(n):
               for j in xrange(i+1,n):
                   mean = (A[i,j]+A[j,i])/2
                   A[i,j] = mean
                   A[j,i] = mean
        return A

    def ADMM_theta_hattheta(self):
        ## update theta
        Q = self.Q
        Uk = self.U1
        LAMBDA,D = np.linalg.eigh(2*self.rho*(Q-Uk)-self.m*self.S)
        D = np.matrix(D)
        theii = (LAMBDA+np.sqrt(LAMBDA**2+8*self.rho*self.m))/(4*self.rho)
        theta_update = np.dot(np.dot(D,np.diag(theii)),D.T)
        assert self.check_symmetric(theta_update)
        self.theta = theta_update

        ## update hat_theta
        hatQ = self.hatQ
        Uk = self.U3
        LAMBDA,D = np.linalg.eigh(2*self.rho*(hatQ-Uk)-self.hatm*self.hatS)
        D = np.matrix(D)
        theii = (LAMBDA+np.sqrt(LAMBDA**2+8*self.rho*self.hatm))/(4*self.rho)
        theta_update = np.dot(np.dot(D,np.diag(theii)),D.T)
        assert self.check_symmetric(theta_update)
        self.hat_theta = theta_update
    def update_T(self):
        Q = self.Q
        theta = self.theta
        U1 = self.U1
        U2 = self.U2
        ## update Q
        t1 = Q-np.dot(U2,Q)+theta+U1
        t2 = np.linalg.inv(np.dot(Q,Q)+np.eye(self.nw))
        RHS = np.dot(t1,t2)
        self.T = self.prox_positive_definite(self.proj2Symmetric(RHS))

    def update_hatT(self):

        Q = self.hatQ
        theta = self.hat_theta
        U3 = self.U3
        U4 = self.U4
        ## update Q
        t1 = Q-np.dot(U4,Q)+theta+U3
        t2 = np.linalg.inv(np.dot(Q,Q)+np.eye(self.nw))
        RHS = np.dot(t1,t2)
#            self.hatT = RHS
        self.hatT = self.prox_positive_definite(self.proj2Symmetric(RHS))
    def update_V(self):
        self.V = self.rho*(self.Q-self.hatQ-self.U5)/(2*self.lamb+self.rho)

    def update_Q(self):
        I = np.eye(self.nw)
        # update Q
        hatQ = self.hatQ
        T = self.T
        t1 = np.linalg.inv(T+I)
        t2 = hatQ+self.U5+self.V-self.U2+I
#            t2 = I
        self.Q = np.dot(t1,t2)

    def update_hatQ(self):

        I = np.eye(self.nw)
        # update Q
        Q = self.Q
        T = self.hatT
        t1 = np.linalg.inv(T+I)
        t2 = Q-self.U5-self.V-self.U4+I
        self.hatQ = np.dot(t1,t2)


    def check_symmetric(self,a, tol=1e-8):
        return np.allclose(a, np.transpose(a), atol=tol)


    def update_U(self):
        I = np.eye(self.nw)
        self.U1 = self.U1+self.theta-self.T
        self.U2 = self.U2 + np.dot(self.T,self.Q)-I
        self.U3 = self.U3+self.hat_theta-self.hatT
        self.U4 = self.U4 + np.dot(self.hatT,self.hatQ)-I
        self.U5 = self.U5 + self.V-self.Q+self.hatQ


    # Returns True if convergence criteria have been satisfied
    def CheckConvergence(self, V_pre, Q_pre,hatQ_pre,T_pre,hatT_pre, e_abs, e_rel, verbose):
        norm = np.linalg.norm
        I = np.eye(self.nw)
        r2 = norm(self.theta-self.T)
        hatr2 = norm(self.hat_theta-self.hatT)
        r1 = norm(np.dot(self.T,self.Q)-I)
        hatr1 = norm(np.dot(self.hatT,self.hatQ)-I)
        r = np.sqrt(r1**2+hatr1**2+r2**2+hatr2**2)

        s1 = norm(T_pre-self.T)
        hats1 = norm(hatT_pre-self.hatT)
        s2 = norm(np.dot(self.T,Q_pre-self.Q))
        hats2 = norm(np.dot(self.hatT,hatQ_pre-self.hatQ))
        s = self.rho*np.sqrt(s1**2+hats1**2+s2**2+hats2**2)

        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = self.nw**2 * e_abs + e_rel * max(np.sqrt(np.linalg.norm(self.theta)**2+norm(self.hat_theta)**2), np.sqrt(np.linalg.norm(self.T)**2+norm(self.hatT)**2),np.sqrt(np.linalg.norm(self.Q)**2+norm(self.hatQ)**2),norm(self.V)) + .01
        e_dual = self.nw**2 * e_abs + e_rel * self.rho * (np.sqrt( norm(self.U1)**2+norm(self.U2)**2+norm(self.U3)**2+norm(self.U4)**2+norm(self.U5)**2))+0.01
        # Primal and dual residuals
        res_pri = r
        res_dual = s
        obj = self.obj_overall()
        if(obj<self.minObj):
            self.minObj = obj
            self.minTheta = self.theta
            self.minHatTheta = self.hat_theta
        self.plist.append(r)
        self.dlist.append(s)
        self.eprilist.append(e_pri)
        self.edualist.append(e_dual)
        if verbose:
            # Debugging information to print convergence criteria values

            print '  r:', res_pri
            print '  e_pri:', e_pri
            print '  s:', res_dual
            print '  e_dual:', e_dual
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)


        return (stop, res_pri, e_pri, res_dual, e_dual)

    #solve
    def __call__(self, maxIters, eps_abs, eps_rel, verbose):
#        print '\n Model 6 ADMM solver: lambdaADMM,nw =', self.lamb,self.nw
        self.status = 'Incomplete: max iterations reached'
        stop = False
        t1 = time.time()
        for i in range(maxIters):
            self.iter = i
            T_old = copy.deepcopy(self.T)
            hatT_old = copy.deepcopy(self.hatT)
            Q_old = copy.deepcopy(self.Q)
            hatQ_old = copy.deepcopy(self.hatQ)
            V_old = copy.deepcopy(self.V)
            try:
                self.ADMM_theta_hattheta()
                self.update_T()
                self.update_hatT()
                self.update_V()
                self.update_Q()
                self.update_hatQ()
                self.update_U()
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    print 'Encounter LinAlgError: Singular matrix, exit ADMM'
                    break
                else:
                    raise



#
            if i>=maxIters-1:
                print 'Incomplete: max iterations reached', i
            if i != 0:
                stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(V_old,Q_old,hatQ_old,T_old,hatT_old, eps_abs, eps_rel, verbose)
                if stop:
                    self.status = 'Optimal'
#                    print "Admm stop early at Iteration ",i
                    if verbose:
                        print "Admm stop early at Iteration ",i
                        print '  r:', res_pri
                        print '  e_pri:', e_pri
                        print '  s:', res_dual
                        print '  e_dual:', e_dual
                    break
        t2 = time.time()
        avgIterTime = (t2-t1)/(self.iter+1)
        if self.nw>=50:
            saveTime = open('data/KDD/efficiency_nw/model7Time_nw'+str(self.nw)+'.txt','a')
            saveTime.write(str(avgIterTime)+' ')
            saveTime.close()
        if not stop:
            self.theta = self.minTheta
            self.hat_theta = self.minHatTheta
        retVal = np.zeros([self.length,2])
        result = np.asmatrix(self.theta[np.triu_indices(self.nw)]).T.getA()
        hatresult = np.asmatrix(self.hat_theta[np.triu_indices(self.nw)]).T.getA()
        retVal[:,0] = np.reshape(result,(self.length,))
        retVal[:,1] = np.reshape(hatresult,(self.length,))

        return retVal





























# -*- coding: utf-8 -*-

