# training function
import numpy as np
import main_OCMFH as mOCMFH
import hammingdist as hd
import map_rank as mr
import inputmatrices
import constants as c

"""
    X*_train : [ num_samples x features ]
    *_te : features x num_samples
    L_t* : num_samples x label_size
"""

def dumpToNpy(data, filename):
    np.save(filename, data)

def loadFromNpy(filename):
    return np.load(filename, allow_pickle = True)[()]

if __name__ == '__main__':
    #X1_train = [ np.array([ [1.0, 1.1, 2.1], [2.0, 2, 1]]), np.array([[1.2, 2.2, 2.9], [9.1, 8.1, 0.1]]) ]
    #X2_train = [ np.array([ [2.0, 6.1], [5.0, 6]]), np.array([[1.7, 1.2], [2.1, 2.1]]) ]
    #I_te = np.array([[1.0, 3.4, 5.2], [2.3, 1.4, 5.8]]).T
    #T_te = np.array([[1.4, 4.4], [2.5, 5.1]]).T
    #L_te = np.array([[0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    #L_tr = np.array([[0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0, 0], [0, 1.0, 0]]) 

    # streamdata,nstream,L_tr,I_tr,T_tr
    #data = inputmatrices.NUS_WIDE(c.dirpath_xv, c.dirpath_xt, c.dirpath_y)

    #print('dumping X1_train...')
    #dumpToNpy(data.X1_train, 'X1_train.npy')
    #print('dumping X2_train...')
    #dumpToNpy(data.X2_train, 'X2_train.npy')
    #print('dumping remaining data...')
    #dumpToNpy(data.X1_test, 'I_te.npy')
    #dumpToNpy(data.X2_test, 'T_te.npy')
    #dumpToNpy(data.Y_test, 'L_te.npy')
    #dumpToNpy(data.Y_train, 'L_tr.npy')
    #exit(0)

    print('Loading data from .npy files...')
    temp_X1_train = loadFromNpy('X1_train.npy')
    temp_X2_train = loadFromNpy('X2_train.npy')
    I_te = loadFromNpy('I_te.npy').T
    T_te = loadFromNpy('T_te.npy').T
    L_te = loadFromNpy('L_te.npy')
    L_tr = loadFromNpy('L_tr.npy')
 
    X1_train = [np.array(temp_X1_train[i]) for i in range(len(temp_X1_train))]
    X2_train = [np.array(temp_X2_train[i]) for i in range(len(temp_X2_train))]
   
    # Learn OCMFH
    print('OCMFH learning started...')
    B_I, B_T, tB_I, tB_T = mOCMFH.main_OCMFH(X1_train, X2_train, I_te, T_te, c.bit)
    
    #  Ground Truth
    #GT = np.matmul(L_te,L_tr.T)
    #GT = np.where(GT>0, 1, 0)

    # compute mAP
    print('computing mAP...')
    itot_hamming_dist = hd.hammingdist(tB_I, B_T)
    ttoi_hamming_dist = hd.hammingdist(tB_T, B_I)
    itot_hamming_rank = np.argsort(itot_hamming_dist, axis = 1)
    ttoi_hamming_rank = np.argsort(ttoi_hamming_dist, axis = 1)
    mAP_itot = mr.map_rank(L_tr,L_te, itot_hamming_rank.T)
    mAP_ttoi = mr.map_rank(L_tr, L_te, ttoi_hamming_rank.T)
    print('image to text mAP@max: \n', np.max(mAP_itot), np.argmax(mAP_itot))
    print('text to image mAP@max: \n', np.max(mAP_ttoi), np.argmax(mAP_ttoi))
    # map = [mapIT(100),mapTI(100)]

