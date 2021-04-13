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

def mysolveOCMFH(self, Itrain, Ttrain, WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj,lambda_, mu, gamma, numiter):
    bits = size(WI,2)
    H = (lambda_ * WI.T * WI + (1- lambda_) * WT.T * WT + 2 * mu * np.eye(bits) + gamma * eye(bits)) / (lambda_ * WI.T * Itrain + (1 - lambda_) * WT.T * Ttrain + mu * (PI * Itrain + PT * Ttrain))
    Uold = [lambda_*WI (1-lambda_)*WT]
    # check Uold
    # Update Parameters
    for i in ranfe(numiter):

        # update U1 and U2
        W1 = W1 + Itrain * H.T
        H1 = H1 + H * H.T

        W2 = W2 + Ttrain * H.T
        H2 = H2 + H * H.T

        WI = W1 / H1
        WT = W2 / H2


        # update V
        H = (lambda_ * WI.T * WI + (1- lambda_) * WT.T * WT + 2 * mu * np.eye(bits) + gamma * np.eye(bits)) / (lambda_ * WI.T * Itrain + (1 - lambda_) * WT.T * Ttrain + mu * (PI * Itrain + PT * Ttrain))

        # % update P1 and P2
        F1 = F1 + H*Itrain.T
        G1 = G1 + Itrain*Itrain.T

        F2 = F2 + H*Ttrain.T
        G2 = G2 + Ttrain*Ttrain.T

        PI = F1 / G1
        PT = F2 / G2

        # Compute object function
        norm1 = self.lambda_ * np.linalg.norm(X1 - U1 * Y, ord='fro')
        norm2 = (1 - self.lambda_) * np.linalg.norm(X2 - U2 * Y, ord='fro')
        norm3 = self.mu * np.linalg.norm(Y - P1 * X1, ord='fro')
        norm4 = self.mu * np.linalg.norm(Y - P2 * X2, ord='fro')
        norm5 = self.gamma * (np.linalg.norm(U1, ord='fro') + np.linalg.norm(U2, ord='fro') + np.linalg.norm(Y,ord='fro') + np.linalg.norm(P1, ord='fro') + np.linalg.norm(P2, ord='fro'))
        currentF = norm1 + norm2 + norm3 + norm4 + norm5
        obj.append(currentF)
        # update HH
        # check Unew
        Unew = [lambda_*WI, (1-lambda_)*WT]
        HH = (Unew.T * Unew + gamma * np.eye(bits)) / (Unew.T * Uold * HH)
        HH.append(H)

    return WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj




