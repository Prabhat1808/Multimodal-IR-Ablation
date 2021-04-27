# training function
import numpy as np
import main_OCMFH as mOCMFH
import hammingdist as hd
import map_rank
import inputmatrices
import constants as c

"""
    X*_train : [ num_samples x features ]
    *_te : features x num_samples
    L_t* : num_samples x label_size
"""

if __name__ == '__main__':
    # streamdata,nstream,L_tr,I_tr,T_tr
    #data = inputmatrices.NUS_WIDE(c.dirpath_xv, c.dirpath_xt, c.dirpath_y)
    #data.stats()

    X1_train = [ np.array([ [1.0, 1.1, 2.1], [2.0, 2, 1]]), np.array([[1.2, 2.2, 2.9], [9.1, 8.1, 0.1]]) ]
    X2_train = [ np.array([ [2.0, 6.1], [5.0, 6]]), np.array([[1.7, 1.2], [2.1, 2.1]]) ]
    I_te = np.array([[1.0, 3.4, 5.2], [2.3, 1.4, 5.8]]).T
    T_te = np.array([[1.4, 4.4], [2.5, 5.1]]).T
    L_te = np.array([[0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    L_tr = np.array([[0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0, 0], [0, 1.0, 0]])
    
    # Learn OCMFH
    B_I, B_T, tB_I, tB_T = mOCMFH.main_OCMFH(X1_train, X2_train, I_te, T_te, c.bit)
    
    #  Ground Truth
    GT = np.matmul(L_te,L_tr.T)
    GT = np.where(GT>0, 1, 0)

    # compute mAP
    itot_hamming_dist = hd.hammingdist(tB_I, B_T)
    ttoi_hamming_dist = hd.hammingdist(tB_T, B_I)
    # HammingRank = np.sort(Dhamm,axis=0)
    # mapIT = map_rank(L_tr,L_te,HammingRank)
    # Dhamm = hammingDist(tB_T, B_I).T
    # HammingRank = np.sort(Dhamm,axis=0)
    # mapTI = map_rank(L_tr,L_te,HammingRank)
    # map = [mapIT(100),mapTI(100)]

