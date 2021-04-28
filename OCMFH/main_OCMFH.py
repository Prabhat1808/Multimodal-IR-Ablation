import numpy as np
import mysolveCMFH as CMFH
import mysolveOCMFH as OCMFH
import hammingdist

"""
    X1_train : [ nsamples x features ] for train images
    X2_train : [ nsamples x features ] for train text
    I_te : features x nsamples for test images
    T_te : features x nsamples fro test text
"""


# return [B_Ir,B_Tr,B_Ie,B_Te,obj,traintime,testtime]
def main_OCMFH(X1_train, X2_train, I_te, T_te, bits, lambda_=0.5, mu=100, gamma=0.001, iter=10, cmfhiter=100):
    nstream = len(X1_train)     #TODO: 0 or 1 ?
    # Initialization ....check and add 
    Itrain = X1_train[0]
    Ttrain = X2_train[0]
    numdata = Itrain.shape[0]
    mean_I = np.mean(Itrain,axis=0) #1 ALgo 3
    mean_T = np.mean(Ttrain,axis=0)
    Itrain = Itrain - mean_I
    Ttrain = Ttrain - mean_T
#    mean_T = mean_T.T #TODO : why transpose
 #   mean_I = mean_I.T #TODO: why transpose
    print("Batch: 1  Total: ", nstream)
    WI, WT, PI, PT, HH, obj = CMFH.mysolveCMFH(Itrain.T, Ttrain.T, lambda_, mu, gamma, cmfhiter, bits) # 2 Algo 3

    # Train---2 to n

    mFea1 = Itrain.shape[1]
    mFea2 = Ttrain.shape[1]
    W1 = np.matmul(Itrain.T, HH.T)
    W2 = np.matmul(Ttrain.T, HH.T)
    H1 = np.matmul(HH, HH.T)
    H2 = H1.copy()
    F1 = np.matmul(HH,Itrain)
    F2 = np.matmul(HH,Ttrain)
    G1 = np.matmul(Itrain.T,Itrain) + gamma*np.eye(mFea1)
    G2 = np.matmul(Ttrain.T,Ttrain) + gamma*np.eye(mFea2)

    for i in range(1,nstream):
        print('stream: {} is running...'.format(i))
        # Initialization ....check and add
        Itrain = X1_train[i]
        Ttrain = X2_train[i]
        numdata_tmp = Itrain.shape[0]  #TODO: shouldn't it be shape[0]
        mean_Itmp = np.mean(Itrain,axis=0) # 4.1 Algo 3
        mean_Ttmp = np.mean(Ttrain,axis=0)
        mean_I = (1 / (numdata + numdata_tmp)) * np.add(numdata * mean_I, numdata_tmp*mean_Itmp)
        mean_T = (1 / (numdata + numdata_tmp)) * np.add(numdata * mean_T, numdata_tmp*mean_Ttmp)
        Itrain = np.subtract(Itrain,mean_I)
        Ttrain = np.subtract(Ttrain, mean_T)
        numdata = numdata + numdata_tmp
        #  4.3 Algo3
        WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj =  OCMFH.mysolveOCMFH(Itrain.T, Ttrain.T, WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj, lambda_, mu, gamma, iter)
         
    Y_tr =np.sign(np.subtract( HH, np.mean(HH, axis = 1).reshape((-1, 1))).T)  # returns sign of each element
    Y_tr[Y_tr<0] = 0

    #B_Tr = compactbit(Y_tr) #TODO: use it if able to pythonize compactbit function
    B_Tr = Y_tr.copy()
    B_Ir = B_Tr.copy()

    # Testing
    Yi_te  = np.sign(np.subtract(np.matmul(PI, I_te), np.mean(HH, axis=1).reshape((-1, 1)))).T  # returns sign
    Yt_te  = np.sign(np.subtract(np.matmul(PT, T_te), np.mean(HH, axis=1).reshape((-1, 1)))).T  # returns sign
    Yi_te[Yi_te < 0] = 0
    Yt_te[Yt_te < 0] = 0
    
    B_Te = Yt_te.copy()
    B_Ie = Yi_te.copy()

    #B_Te = compactbit(Yt_te) #TODO: use it if able to pythonize compactbit function
    #B_Ie = compactbit(Yi_te) #TODO: use it if able to pythonize compactbit function

    return B_Ir, B_Tr, B_Ie, B_Te
