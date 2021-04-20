# training function
import numpy as np
import main_OCMFH
import hammingdist
import map_rank
import inputmatrices

if __name__ == '__main__':
    # streamdata,nstream,L_tr,I_tr,T_tr
    #  Read data
    #  Ground Truth
    GT = np.matmul(L_te,L_tr.T)
    WtrueTestTraining = np.zeros(L_te.shape[0],L_tr.shape[0])
    for i in range(GT.shape[0]):
        for j in range(GT.shape[1]):
            if GT[i][j]>0:
                WtrueTestTraining[i][j]=1

    # Parameter setting
    bit = 32

    # Learn OCMFH
    B_I, B_T, tB_I, tB_T = main_OCMFH(streamdata, I_te, T_te, bit)

    # compute mAP
    # Dhamm = hammingdist(tB_I, B_T)
    # HammingRank = np.sort(Dhamm,axis=0)
    # mapIT = map_rank(L_tr,L_te,HammingRank)
    # Dhamm = hammingDist(tB_T, B_I).T
    # HammingRank = np.sort(Dhamm,axis=0)
    # mapTI = map_rank(L_tr,L_te,HammingRank)
    # map = [mapIT(100),mapTI(100)]

