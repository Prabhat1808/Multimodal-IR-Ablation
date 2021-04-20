import numpy as np
import mysolveCMFH
import hammingdist

# return [B_Ir,B_Tr,B_Ie,B_Te,obj,traintime,testtime]
def main_OCMFH(streamdata, I_te, T_te, bits, lambda_, mu, gamma, iter, cmfhiter):
    if lambda_ == None:
        lambda_ = 0.5
    if mu == None:
        mu = 100
    if gamma == None:
        gamma = 0.001
    if iter == None:
        iter = 10
    if cmfhiter == None:
        cmfhiter = 100
    nstream = streamdata.shape[1]
    # Initialization ....check and add 
    Itrain = streamdata
    Ttrain = streamdata
    mean_I = np.mean(Itrain,axis=1)# 1 ALgo 3
    mean_T =np.mean(Ttrain,axis=1)
    Itrain= np.subtract(Itrain, mean_I)
    Ttrain = np.subtract(Ttrain, mean_T)
    mean_T = mean_T.T
    mean_I = mean_I.T
    print("Batch: 1  Total: ", nstream)
    WI, WT, PI, PT, HH, Obj = mysolveCMFH(Itrain.T, Ttrain.T, lambda_, mu, gamma, cmfhiter, bits) # 2 Algo 3

    # Train---2 to n

    mFea1 = Itrain.shape[1]
    mFea1 = Ttrain.shape[1]
    W1 = np.matmul(Itrain.T, HH.T)
    W2 = np.matmul(Ttrain.T, HH.T)
    H1 = np.matmul(HH, Itrain)
    H2 = np.matmul(HH, Ttrain)
    H2=H1
    F1 = np.matmul(HH,Itrain)
    F2 = np.matmul(HH,Ttrain)
    G1 = np.add(np.matmul(Itrain.T,Itrain),gamma*np.eye(mFea1)) 
    G2 = np.add(np.matmul(Ttrain.T,Ttrain),gamma*np.eye(mFea2))

    for i in range(2,nstream):
        # Initialization ....check and add
        Itrain = streamdata
        Ttrain = streamdata
        numdata_tmp = Itrain.shape[1]
        mean_Itmp = np.mean(Itrain,axis=1) # 4.1 Algo 3
        mean_Ttmp = np.mean(Ttrain,axis=1)
        mean_I = (1 / numdata + numdata_tmp) * np.add(numdata * mean_I, numdata_tmp*mean_Itmp)
        mean_T = (1 / numdata + numdata_tmp) * np.add(numdata * mean_T, numdata_tmp*mean_Ttmp)
        Itrain = np.subtract(Itrain,mean_I)
        Ttrain = np.subtract(Ttrain, mean_T)
        numdata = numdata + numdata_tmp
        #  4.3 Algo3
        WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj =  mysolveOCMFH(Itrain, Ttrain, WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj, lambda_, mu, gamma, iter)
         
    Y_tr =np.sign(np.subtract( HH, np.mean(HH, axis = 1)).T)  # returns sign of each element
    for i in range(Y_tr.shape[0]):
        if Y_tr[i]<0:
            Y_tr = 0

    B_Tr = compactbit(Y_tr)
    B_Ir = B_Tr

    # Testing
    Yi_te  = np.sign(np.subtract(np.matmul(PI, I_te), np,mean(HH, axis=1)).T)  # returns sign
    Yt_te  = np.sign(np.subtract(np.matmul(PT, T_te), np,mean(HH, axis=1)).T)  # returns sign
    for i in range(Yi_te.shape[0]):
        if Yi_te[i]<0:
            Yi_te = 0

    for i in range(Yt_te.shape[0]):
        if Yt_te[i]<0:
            Yt_te=0

    B_Te = compactbit(Yt_te)
    B_Ie = compactbit(Yi_te)

