#For ICDM review only, please do not distribute
#ALL RIGHTS RESERVED
#ADMM solver for CPM-C model with parital correlation based translation function
import numpy as np
import copy
import time

class myADMMSolver:
    def __init__(self,lamb, nw, rho,x,hat_x,rho_update_func=None):
        self.rho = rho
        self.lamb = lamb
        self.length = nw*(nw+1)/2  #vector length of trianglular matrix
        self.S = np.cov(np.transpose(x))
        self.hatS = np.cov(np.transpose(hat_x))
        self.nw = nw;
        self.m = x.shape[0]
        self.hatm = hat_x.shape[0]
        self.theta = np.zeros((nw,nw))
        self.hat_theta = np.zeros((nw,nw))
        self.Q= np.zeros((nw,nw))
        self.hatQ = np.zeros((nw,nw))
        self.T =np.zeros((nw,nw))   # \Gamma in the paper
        self.hatT = np.zeros((nw,nw))  # \hat \Gamma
        self.P= np.zeros((nw,nw))
        self.hatP = np.zeros((nw,nw))
        self.V = np.zeros((nw,nw))
        self.Y = np.zeros((nw,nw))
        self.hatY = np.zeros((nw,nw))
        self.Z = np.zeros((nw,nw))  #  theta & T
        self.hatZ = np.zeros((nw,nw))
        self.U = np.zeros((nw,nw))   # P & V
        self.plist = []
        self.dlist = []
        self.eprilist = []
        self.edualist = []
        self.objlist = []


# In[]   help/debug methods
    def computePartialCorrelation(self, T):
        P = np.ones(T.shape)
        nw = T.shape[0]
        for i in range(nw):
            for j in range(i+1,nw):
                P[i,j] = -T[i,j]/np.sqrt(T[i,i]*T[j,j])
                P[j,i] = P[i,j]
        return P
    def h(self,theta,S,m):
        return 0.5*m*(np.trace(np.dot(S,theta))-np.log(np.linalg.det(theta)))
    def obj_overall(self):
        V = self.computePartialCorrelation(self.T)-self.computePartialCorrelation(self.hatT)
        return self.h(self.T,self.S,self.m)+self.h(self.hatT,self.hatS,self.hatm)+self.lamb*(np.linalg.norm(V)**2)

    def check_symmetric(self,a, tol=1e-3):
        return np.allclose(a, np.transpose(a), atol=tol)


# In[]  update variables
    def update_T(self):
        LAMBDA,D = np.linalg.eigh(2*self.rho*(self.theta-self.Z)-self.m*self.S)
        D = np.matrix(D)
        theii = (LAMBDA+np.sqrt(LAMBDA**2+8*self.rho*self.m))/(4*self.rho)
        self.T = np.dot(np.dot(D,np.diag(theii)),D.T)
#        self.objT.append(self.objective_T())
    def update_hatT(self):
        LAMBDA,D = np.linalg.eigh(2*self.rho*(self.hat_theta-self.hatZ)-self.hatm*self.hatS)
        D = np.matrix(D)
        theii = (LAMBDA+np.sqrt(LAMBDA**2+ 8*self.rho*self.hatm))/(4*self.rho)
        self.hatT = np.dot(np.dot(D,np.diag(theii)),D.T)
#        self.objhatT.append(self.objective_hatT())
    def update_theta(self):
        theta = np.eye(self.nw)
        for i in range(self.nw):
            theta[i,i] = self.T[i,i]+self.Z[i,i]
        for i in range(self.nw):
            for j in range(i+1,self.nw):
                c = 1/np.sqrt(theta[i,i]*theta[j,j])
                theta[i,j] = (self.T[i,j]+self.Z[i,j]- c*self.P[i,j]-c*self.Y[i,j])/(c**2+1)
                theta[j,i] = theta[i,j]
        self.theta = theta
        assert self.check_symmetric(self.theta)
        self.Q = self.computePartialCorrelation(self.theta)

    def update_hat_theta(self):
        hat_theta = np.eye(self.nw)
        for i in range(self.nw):
            hat_theta[i,i] = self.hatT[i,i]+self.hatZ[i,i]
        for i in range(self.nw):
            for j in range(i+1,self.nw):
                c = 1/np.sqrt(hat_theta[i,i]*hat_theta[j,j])
                hat_theta[i,j] = (self.hatT[i,j]+self.hatZ[i,j]-c*self.hatP[i,j]-c*self.hatY[i,j])/(1+c**2)
                hat_theta[j,i] = hat_theta[i,j]
        self.hat_theta = hat_theta
        assert self.check_symmetric(self.hat_theta)
        self.hatQ = self.computePartialCorrelation(self.hat_theta)

    def update_V(self):
        self.V = self.rho*(self.P-self.hatP-self.U)/(2*self.lamb+self.rho)
        assert self.check_symmetric(self.V)
        assert np.linalg.norm(np.diag(self.V))==0
    def proj2Symmetric(self,A):
        n  = A.shape[0]
        for i in xrange(n):
               for j in xrange(i+1,n):
                   mean = (A[i,j]+A[j,i])/2
#                   if mean<0:
#                       mean = 0
                   A[i,j] = mean
                   A[j,i] = mean
        return A

    def update_P(self):
        self.P = (self.V+self.hatP+self.U+self.Q-self.Y)/2
        for i in range(self.nw):
            self.P[i,i] = 1
        assert self.check_symmetric(self.P)

    def update_hatP(self):
        self.hatP = (self.P-self.V-self.U+self.hatQ-self.hatY)/2
        for i in range(self.nw):
            self.hatP[i,i] = 1
        assert self.check_symmetric(self.hatP)
#

    def update_duals(self):
        self.Y = self.P-self.Q+self.Y
        self.hatY = self.hatP-self.hatQ+self.hatY
        self.Z = self.T-self.theta+self.Z
        self.hatZ = self.hatT-self.hat_theta+self.hatZ

        self.U = self.V-self.P+self.hatP+self.U


    def CheckConvergence(self,Q_pre,hatQ_pre, theta_pre,hat_theta_pre,V_pre,P_pre,hatP_pre, e_abs, e_rel, verbose):
        r1 = self.T -self.theta
        r2 = self.hatT-self.hat_theta
        r3 = self.V-self.P+self.hatP
        r4 = self.P-self.Q
        r5 = self.hatP-self.hatQ
        allR = np.concatenate((r1,r2,r3,r4,r5))
        norm = np.linalg.norm
        r = norm(allR)
        s1 = self.Q-Q_pre
        s2 = self.hatQ-hatQ_pre
        s3 = self.V-V_pre
        s4 = self.P-P_pre
        s5 = self.hatP - hatP_pre
        allS = np.concatenate((s1,s2,s3,s4,s5))*self.rho

        s = norm(allS)

        e_pri = self.nw * e_abs + e_rel * max(norm(self.theta),norm(self.hat_theta), norm(self.T)+norm(self.hatT),norm(self.P),norm(self.hatP),norm(self.V))
        e_dual = np.sqrt((self.nw**2)) * e_abs + e_rel *  (np.sqrt(self.rho *( norm(self.Z)**2+norm(self.hatZ)**2+norm(self.Y)**2+norm(self.hatY)**2+norm(self.U)**2)))
        res_pri = r
        res_dual = s

        self.plist.append(r)
        self.dlist.append(s)
        self.eprilist.append(e_pri)
        self.edualist.append(e_dual)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)



        return (stop, res_pri, e_pri, res_dual, e_dual)

    # solve
    def __call__(self, eps_abs, eps_rel, verbose,admmMaxIters=1000):
#        print '\n solver ADMM model 7: lambdaADMM = ',self.lamb
        self.status = 'Incomplete: max iterations reached'
        t1 = time.time()
        for i in range(admmMaxIters):
            self.iter = i
            Q_pre = copy.deepcopy(self.Q)
            hatQ_pre = copy.deepcopy(self.hatQ)
            theta_pre = copy.deepcopy(self.theta)
            hat_theta_pre = copy.deepcopy(self.hat_theta)
            P_pre = copy.deepcopy(self.P)
            hatP_pre = copy.deepcopy(self.hatP)
            V_pre = copy.deepcopy(self.V)
            try:
                self.update_T()
                self.update_hatT()
                self.update_theta()
                self.update_hat_theta()
                self.update_V()
                self.update_P()
                self.update_hatP()
                self.update_duals()
                self.objlist.append(self.obj_overall())
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    print 'Encounter LinAlgError: Singular matrix, exit ADMM'
                    break
                else:
                    raise



#
            if i>=admmMaxIters-1:
                print 'Incomplete: max iterations reached', i
            if i != 0:
                stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(Q_pre,hatQ_pre, theta_pre,hat_theta_pre,V_pre,P_pre,hatP_pre, eps_abs, eps_rel, verbose)
                if stop:
                    self.status = 'Optimal'
                    if verbose:
                        print "Admm stop early at Iteration ",i
                        print '  r:', res_pri
                        print '  e_pri:', e_pri
                        print '  s:', res_dual
                        print '  e_dual:', e_dual
                    break
#
                new_rho = self.rho
                threshold = 10
                if (res_pri>threshold*res_dual):
                    new_rho = 2*self.rho
                elif (threshold*res_pri<res_dual):
                    new_rho = self.rho/2.0
                scale = self.rho / new_rho
                self.rho = new_rho
                self.U = scale*self.U
                self.Y = scale*self.Y
                self.hatY = scale*self.hatY
                self.Z = scale*self.Z
                self.hatZ = scale*self.hatZ

        t2 = time.time()
        avgIterTime = (t2-t1)/(self.iter+1)
#        print " avgPerADMMIterTime",avgIterTime
        if self.nw>=50:
            saveTime = open('data/KDD/efficiency_nw/model6Time_nw'+str(self.nw)+'.txt','a')
            saveTime.write(str(avgIterTime)+' ')
            saveTime.close()
        retVal = np.zeros([self.length,2])
        result = np.asmatrix(self.T[np.triu_indices(self.nw)]).T.getA()
        hatresult = np.asmatrix(self.hatT[np.triu_indices(self.nw)]).T.getA()
        retVal[:,0] = np.reshape(result,(self.length,))
        retVal[:,1] = np.reshape(hatresult,(self.length,))
        return retVal




















































