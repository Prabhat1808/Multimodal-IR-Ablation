'''
function [ WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj] = mysolveOCMFH(Itrain, Ttrain, WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj,lambda, mu, gamma, numiter)
% Reference:
% Di Wang, Quan Wang, Yaqiang An, Xinbo Gao, and Yumin Tian.
% Online Collective Matrix Factorization Hashing for Large-Scale Cross-Media Retrieval.
% In 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'20), July 25¨C30, 2020, Virtual Event,
% China. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3397271.3401132
% (Manuscript)
%
% Version1.0 -- Jan/2020
% Contant: Di Wang (wangdi@xidain.edu.cn)
%
'''

import numpy as np

def mysolveOCMFH(Itrain, Ttrain, WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj, lambda_, mu, gamma, numiter):
    bits = WI.shape[1]
    # eqution 22
    H = np.matmul(\
            np.linalg.inv((lambda_ * np.matmul(WI.T, WI) + (1- lambda_) * np.matmul(WT.T, WT) + (2 * mu + gamma) * np.eye(WI.shape[1]))),\
            (lambda_ * np.matmul(WI.T, Itrain) + (1 - lambda_) * np.matmul(WT.T, Ttrain) + mu * (np.matmul(PI, Itrain) + np.matmul(PT, Ttrain)))\
        )
    # update HH
    Uold = np.concatenate((lambda_*WI, (1-lambda_)*WT), axis=0)
    # Update Parameters
    for i in range(numiter):
        # update U1 and U2
        W1 = W1 + np.matmul(Itrain, H.T) # equation 10
        H1 = H1 + np.matmul(H, H.T) # equation 11

        W2 = W2 + np.matmul(Ttrain, H.T) # equation 13
        H2 = H2 + np.matmul(H, H.T) # equation 11

        WI = np.matmul(W1, np.linalg.inv(((H1 + (gamma/lambda_)*np.eye(H1.shape[0]))))) # equation 9
        WT = np.matmul(W2, np.linalg.inv(((H2 + (gamma/lambda_)*np.eye(H2.shape[0]))))) # equation 12


        # update V
        H = np.matmul(\
                np.linalg.inv(lambda_ * np.matmul(WI.T, WI) + (1- lambda_) * np.matmul(WT.T, WT) + (2 * mu + gamma) * np.eye(WI.shape[1])), \
                (lambda_ * np.matmul(WI.T, Itrain) + (1 - lambda_) * np.matmul(WT.T, Ttrain) + mu * (np.matmul(PI, Itrain) + np.matmul(PT, Ttrain)))
            )
        
        # % update P1 and P2
        F1 = F1 + np.matmul(H, Itrain.T) # equation 15
        G1 = G1 + np.matmul(Itrain, Itrain.T) # equation 16

        F2 = F2 + np.matmul(H, Ttrain.T) # equation 18
        G2 = G2 + np.matmul(Ttrain, Ttrain.T) # equation 19

        PI = np.matmul(F1, np.linalg.inv(G1 + (gamma/mu) * np.eye(G1.shape[0]))) # equation 14
        PT = np.matmul(F2, np.linalg.inv(G2 + (gamma/mu) * np.eye(G2.shape[0]))) # equation 17

        # Compute object function
        # equation 6
        norm1 = lambda_ * np.linalg.norm(Itrain - np.matmul(WI, H), ord='fro')
        norm2 = (1 - lambda_) * np.linalg.norm(Ttrain - np.matmul(WT, H), ord='fro')
        norm3 = mu * np.linalg.norm(H - np.matmul(PI, Itrain), ord='fro')
        norm4 = mu * np.linalg.norm(H - np.matmul(PT, Ttrain), ord='fro')
        norm5 = gamma * (np.linalg.norm(WI, ord='fro') + np.linalg.norm(WT, ord='fro') + \
                np.linalg.norm(H,ord='fro') + np.linalg.norm(PI, ord='fro') + np.linalg.norm(PT, ord='fro'))
        currentF = norm1**2 + norm2**2 + norm3 + norm4 + norm5**2
        obj.append(currentF)
        print('{}th iteration of OCMFH with loss {}...'.format(i, currentF))
    Unew = np.concatenate( (lambda_*WI, (1-lambda_)*WT), axis=0)
    HH = np.matmul(\
            np.linalg.inv( np.matmul(Unew.T, Unew) + gamma * np.eye(Unew.shape[1])),\
            ( np.matmul(np.matmul(Unew.T, Uold), HH))\
         )
    HH = np.concatenate((HH, H), axis=1)
    
    return WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj
