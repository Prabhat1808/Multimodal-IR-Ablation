'''
function [ap] = map_rank(traingnd,testgnd, IX)
%% Label Matrix
if isvector(traingnd)
    traingnd = sparse(1:length(traingnd), double(traingnd), 1); traingnd = full(traingnd);
end
if isvector(testgnd)
    testgnd = sparse(1:length(testgnd), double(testgnd), 1); testgnd = full(testgnd);
end

[numtrain, numtest] = size(IX);
apall = zeros(numtrain,numtest);
aa = 1:numtrain;
for i = 1 : numtest
    y = IX(:,i);
    new_label=zeros(1,numtrain);
    new_label(traingnd*testgnd(i,:)'>0)=1;
    xx = cumsum(new_label(y));
    x = xx.*new_label(y);
    p = x./aa;
    p = cumsum(p);
    id = find(p~=0);
    p(id) = p(id)./xx(id);
    apall(:,i) = p';
end
ap = mean(apall,2);
'''
import numpy as np

def map_rank(traingnd, testgnd, hamming_rank):
    """ 
        This funtion returns map@all metric score.
        hamming_rank : numtrain x numtest
        *gnd : numsamples x labelsize
    """
    numtrain, numtest = hamming_rank.shape
    apall = np.zeros((numtrain, numtest))
    aa = np.array([i+1 for i in range(numtrain)])
    for i in range(numtest):
        y = hamming_rank[:, i]
        new_label = np.array([0 for j in range(numtrain)])
        relevant_indices = (np.matmul(traingnd, testgnd[i, :].reshape((-1, 1))) > 0).reshape(-1)
        new_label[relevant_indices] = 1
        xx = np.cumsum(new_label[y])
        x = xx * new_label[y]
        p = x / aa
        p = np.cumsum(p)
        mask = (p != 0)
        p[mask] = p[mask]/xx[mask]
        apall[:, i] = p.copy()
    mAP = np.mean(apall, axis=1)
    return mAP
















