import numpy as np
from scipy import stats
import __builtin__ as bt


def plot2DCluster(X,ylabel=[],zLabel=[]):

    xPos = X[:,0]
    yPos = X[:,1]
    l = xPos.shape[0]
    clrs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    xmax = np.max(xPos)
    ymax = np.max(yPos)
    xmin = np.min(xPos)
    ymin = np.min(yPos)
    Cluster = np.zeros(l)
    if len(zLabel)>0:
        Z = zLabel
    else:
        Z = np.ones(l)
    for i in xrange(l):
        if Z[i]==0:
            Cluster[i] = 2*ylabel[i]
        else:
            Cluster[i] = 2*ylabel[i]+1
    maxCluster = np.int(np.max(Cluster)+1)
    uniCluster = np.unique(Cluster)
    print "uniCluster = ",uniCluster
    for c in xrange(maxCluster):
        x = xPos[np.where(Cluster==c)]
        y = yPos[np.where(Cluster==c)]
        print 'num of points:',len(x)
  
def plotMultiGaussian(a,mean = [],points = 1000,color="C0"):
    inva = np.linalg.inv(a)
    print "a, inva \n",a
    print inva
    x, y = np.random.multivariate_normal(np.zeros(2), a, 1000).T
   
def oneEntry(theta):
    [m,n] = theta.shape
    res = np.zeros([m,n])
    for i in xrange(m):
        for j in xrange(n):
            if theta[i,j] < 0:
                res[i,j] = -1
            elif theta[i,j]>0:
                res[i,j] = 1


    eigs, _ = np.linalg.eig(res)
    lambda_min = min(eigs)
    res = res+(0.1+abs(lambda_min))*np.identity(m)
    return res
def negLoglikelihood(x,theta,mu=0):
    LOG2PI = np.log(2*np.pi)
    x1 = x-mu
    n = len(x)
    logDETtheta = np.log(np.linalg.det(theta))
    ll = (np.dot(np.dot(x1.T,theta),x1)-logDETtheta+n*LOG2PI)/2
    return ll
def logDET(theta):
    return np.log(np.linalg.det(theta))
def extractSlidingWindow(TS,w,n):
    # TS = R^{m*n}, where m is the length of TS
    # T: the total length/row of extracted sliding window
    # n: number of sensors
    # w: sliding window size
########## example ###################
#w = 4
#n = 3
#mts = np.random.randint(0,10,[6,3])
#X = extractSlidingWindow(mts,w,n)
#print "w,n",w,n
#print "mts:\n",mts
#print "X:\n",X
#w,n 4 3
#mts:
#[[8 2 8]
# [4 3 0]
# [4 3 6]
# [9 8 0]
# [8 5 9]
# [0 9 6]]
#X:
#[[ 8.  2.  8.  4.  3.  0.  4.  3.  6.  9.  8.  0.]
# [ 4.  3.  0.  4.  3.  6.  9.  8.  0.  8.  5.  9.]
# [ 4.  3.  6.  9.  8.  0.  8.  5.  9.  0.  9.  6.]]
################################################
    T = TS.shape[0]-w+1
    X = np.zeros([T,n*w])
    for t in xrange(T):
        for i in xrange(w):
            X[t,i*n:(i+1)*n] = TS[t+i,:]
    return X
def bestMatchCluster(Yassign,clusterGroundTruth):
    num_clusters = np.unique(clusterGroundTruth).shape[0]
    train_confusion_matrix = compute_confusion_matrix1(num_clusters,Yassign,clusterGroundTruth)
    matching = find_matching(train_confusion_matrix)
    clustered_points = []
    for i in xrange(len(Yassign)):
        clustered_points.append(matching[int(Yassign[i])])
    return clustered_points

def computeF1Score(Yassign,clusterGroundTruth, plotCluster = False):
    # plot matched
    num_clusters = np.unique(clusterGroundTruth).shape[0]
#    train_confusion_matrix = compute_confusion_matrix1(num_clusters,Yassign,clusterGroundTruth)
    train_confusion_matrix = compute_confusion_matrix1(num_clusters,clusterGroundTruth,Yassign)

    matching = find_matching(train_confusion_matrix)
    clustered_points = []
    for i in xrange(len(Yassign)):
        clustered_points.append(matching[int(Yassign[i])])
    f1_tr = computeF1_macro(train_confusion_matrix,matching,num_clusters)
    if plotCluster:
        print "plt is disabled"
    return f1_tr
def computeF1ScoreWithoutBestMatch(Yassign,clusterGroundTruth, plotCluster = False):
    # plot matched
    num_clusters = np.unique(clusterGroundTruth).shape[0]
    train_confusion_matrix = compute_confusion_matrix1(num_clusters,Yassign,clusterGroundTruth)
    matching = range(num_clusters)
    clustered_points = Yassign
    f1_tr = computeF1_macro(train_confusion_matrix,matching,num_clusters)
    if plotCluster:
        print "plt is disabled"
        
    return f1_tr

def computeF1Score4Z(num_clusters,clustered_points,clusterGroundTruth):

    train_confusion_matrix = compute_confusion_matrix1(num_clusters,clustered_points,clusterGroundTruth)
    matching = find_matching(train_confusion_matrix)
    f1_tr = computeF1_macro(train_confusion_matrix,matching,num_clusters)

    return f1_tr
def getTrainTestSplit(m, num_blocks, num_stacked):
    '''
    - m: number of observations
    - num_blocks: window_size + 1
    - num_stacked: window_size
    Returns:
    - sorted list of training indices
    '''
    # Now splitting up stuff
    # split1 : Training and Test
    # split2 : Training and Test - different clusters
    training_percent = 1
    # list of training indices
    training_idx = np.random.choice(m-num_blocks+1, size=int((m-num_stacked)*training_percent),replace = False )
    # Ensure that the first and the last few points are in
    training_idx = list(training_idx)
    if 0 not in training_idx:
        training_idx.append(0)
    if m - num_stacked  not in training_idx:
        training_idx.append(m-num_stacked)
    training_idx = np.array(training_idx)
    return sorted(training_idx)

def upperToFull(a, eps = 0):
        ind = (a<eps)&(a>-eps)
        a[ind] = 0
        n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)
        A = np.zeros([n,n])
        A[np.triu_indices(n)] = a
        temp = A.diagonal()
        A = np.asarray((A + A.T) - np.diag(temp))
        return A

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    lv = bt.len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    out = tuple([x/256.0 for x in out])
    return out
def reassignY(lle_allNodes,gamma=1):
    (T,K) = lle_allNodes.shape
    Y = np.zeros(T)
    min_lle = np.amin(lle_allNodes)
    ll0 = lle_allNodes - min_lle
#    print "ll0",ll0
    preCost = np.zeros([K])
    curCost = np.zeros([K])
    #prePath = [[]]*(K*2)       # encoding 10 11 20 21... with 0 1 2 3 4...
    #curPath = [[]]*(K*2)
    pathMat = np.zeros([T,K],dtype = 'int32')
    #
    cost = []
    minNode = 0
    for t in xrange(T):
    #    if t%1000 == 0:
    #        print t

    #topTen = 1000;
    #for t in xrange(topTen):
    #    curPath = [[]]*(K*2)
        pminIdx0 = np.argmin(preCost)
        preMin0 = preCost[pminIdx0]
        for k in xrange(K):
            curCost0 = preCost[k]
            preMinCost0 = preMin0+gamma
            cost = [curCost0,  preMinCost0]
    #        cost = cost+ll0[t,k]
            minNode = np.argmin(cost)
            curCost[k] = cost[minNode]+ll0[t,k]
#            print "Z = 0, cost: ",cost
#            print "Z = 0, minNode: ",minNode
#
            if minNode == 0:
                pathMat[t,k] = k
    #            curPath[k*2] = copy.deepcopy(prePath[k*2])

            elif minNode == 1:
                pathMat[t,k] = pminIdx0
#        print "pathMat",pathMat
    #            curPath[k*2] = copy.deepcopy(prePath[k*2+1])


#            curCost0 = preCost[k,0]
#            preMinCost0 = preMin0+gamma
#            cost = [curCost0,  preMinCost0]
#
#            minNode = np.argmin(cost)

        preCost = curCost
    #    prePath = curPath

#    finalMinIdx = np.argmax(curCost)
    finalMinIdx = np.argmin(curCost)
    #finalPath = curPath[finalMinIdx]
    #print "finalPath : ",finalPath
#        print "finalMinIdx : ",finalMinIdx

    #finalPath1 = []
    curNode = finalMinIdx
#    preNode = finalMinIdx
    Y[T-1] = finalMinIdx
    #finalPath1.append(curNode)
    for t in xrange(T-1,0,-1):
        preNode = pathMat[t,curNode]
        Y[t-1] = preNode
#        if not(np.equal(Y[p][t],old_Y[p][t])):
#            Y_changed_count = Y_changed_count+1

#            if i>1000 and i<1050:
#                print 'Z[i]',i,Z[i]
    #    print "i = {},  node = {}".format(i,preNode)
    #    finalPath1.append(preNode)
        curNode = preNode
    return Y
def updateClusters(LLE_node_vals,switch_penalty = 1):
    """
    Takes in LLE_node_vals matrix and computes the path that minimizes
    the total cost over the path
    Note the LLE's are negative of the true LLE's actually!!!!!

    Note: switch penalty > 0
    """
    print "should not be here"
    (T,num_clusters) = LLE_node_vals.shape
    future_cost_vals = np.zeros(LLE_node_vals.shape)

    ##compute future costs
    for i in xrange(T-2,-1,-1):
        j = i+1
        indicator = np.zeros(num_clusters)
        future_costs = future_cost_vals[j,:]
        lle_vals = LLE_node_vals[j,:]
        for cluster in xrange(num_clusters):
            total_vals = future_costs + lle_vals + switch_penalty
            total_vals[cluster] -= switch_penalty
            future_cost_vals[i,cluster] = np.min(total_vals)

    ##compute the best path
    path = np.zeros(T)

    ##the first location
    curr_location = np.argmin(future_cost_vals[0,:] + LLE_node_vals[0,:])
    path[0] = curr_location

    ##compute the path
    for i in xrange(T-1):
        j = i+1
        future_costs = future_cost_vals[j,:]
        lle_vals = LLE_node_vals[j,:]
        total_vals = future_costs + lle_vals + switch_penalty
        total_vals[int(path[i])] -= switch_penalty

        path[i+1] = np.argmin(total_vals)

    ##return the computed path
    return path

def find_matching(confusion_matrix):
    """
    returns the perfect matching
    """
    _,n = confusion_matrix.shape
    path = []
    for i in xrange(n):
        max_val = -1e10
        max_ind = -1
        for j in xrange(n):
            if j in path:
                pass
            else:
                temp = confusion_matrix[i,j]
                if temp > max_val:
                    max_val = temp
                    max_ind = j
        path.append(max_ind)
    return path

def computeF1Score_delete(num_cluster,matching_algo,actual_clusters,threshold_algo,save_matrix = False):
    """
    computes the F1 scores and returns a list of values
    """
    F1_score = np.zeros(num_cluster)
    for cluster in xrange(num_cluster):
        matched_cluster = matching_algo[cluster]
        true_matrix = actual_clusters[cluster]
        estimated_matrix = threshold_algo[matched_cluster]
        if save_matrix: np.savetxt("estimated_matrix_cluster=" + str(cluster)+".csv",estimated_matrix,delimiter = ",", fmt = "%1.4f")
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in xrange(num_stacked*n):
            for j in xrange(num_stacked*n):
                if estimated_matrix[i,j] == 1 and true_matrix[i,j] != 0:
                    TP += 1.0
                elif estimated_matrix[i,j] == 0 and true_matrix[i,j] == 0:
                    TN += 1.0
                elif estimated_matrix[i,j] == 1 and true_matrix[i,j] == 0:
                    FP += 1.0
                else:
                    FN += 1.0
        precision = (TP)/(TP + FP)
        print "cluster #", cluster
        print "TP,TN,FP,FN---------->", (TP,TN,FP,FN)
        recall = TP/(TP + FN)
        f1 = (2*precision*recall)/(precision + recall)
        F1_score[cluster] = f1
    return F1_score
def compute_confusion_matrix1(num_clusters,clustered_points_algo, groundTruth):
    """
    computes a confusion matrix and returns it
    """
#    seg_len = 400
    true_confusion_matrix = np.zeros([num_clusters,num_clusters])
    for point in xrange(bt.len(clustered_points_algo)):
        cluster = clustered_points_algo[point]
#        num = (int(sorted_indices_algo[point]/seg_len) %num_clusters)
        num = groundTruth[point]
        true_confusion_matrix[int(num),int(cluster)] += 1
    return true_confusion_matrix
def compute_confusion_matrix(num_clusters,clustered_points_algo, sorted_indices_algo):
    """
    computes a confusion matrix and returns it
    """
    seg_len = 400
    true_confusion_matrix = np.zeros([num_clusters,num_clusters])
    for point in xrange(bt.len(clustered_points_algo)):
        cluster = clustered_points_algo[point]
        num = (int(sorted_indices_algo[point]/seg_len) %num_clusters)
        true_confusion_matrix[int(num),int(cluster)] += 1
    return true_confusion_matrix

def computeF1_macro(confusion_matrix,matching, num_clusters):
    """
    computes the macro F1 score
    confusion matrix : requres permutation
    matching according to which matrix must be permuted
    """
    ##Permute the matrix columns
    permuted_confusion_matrix = np.zeros([num_clusters,num_clusters])
    for cluster in xrange(num_clusters):
        matched_cluster = matching[cluster]
        permuted_confusion_matrix[:,cluster] = confusion_matrix[:,matched_cluster]
     ##Compute the F1 score for every cluster
    F1_score = 0
    for cluster in xrange(num_clusters):
        TP = permuted_confusion_matrix[cluster,cluster]
        FP = np.sum(permuted_confusion_matrix[:,cluster]) - TP
        FN = np.sum(permuted_confusion_matrix[cluster,:]) - TP
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
#        print 'cluster',cluster
#        print 'TP', TP
#        print 'FP', FP
#        print 'FN', FN
#        print 'permuted_confusion_matrix\n',permuted_confusion_matrix
#        print 'precision',precision
#        print 'recall',recall
        if TP ==0:
            f1 = 0
        else:
            f1 = stats.hmean([precision,recall])
        F1_score += f1
    F1_score /= num_clusters
    return F1_score

