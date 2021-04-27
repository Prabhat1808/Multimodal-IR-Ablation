'''
Compute hamming distance between two sets of samples (B1, B2)
Dh=hammingDist(B1, B2);

Input
   B1, B2: compact bit vectors. Each datapoint is one row.
   size(B1) = [ndatapoints1, nwords]
   size(B2) = [ndatapoints2, nwords]
   It is faster if ndatapoints1 < ndatapoints2

Output
   Dh = hamming distance.
   size(Dh) = [ndatapoints1, ndatapoints2]

example query
Dhamm = hammingDist(B2, B1);
this will give the same result than:
    Dhamm = distMat(U2>0, U1>0).^2;
the size of the distance matrix is:
   size(Dhamm) = [Ntest x Ntraining]

loop-up table:
'''
import numpy as np

def hammingdist(_B1,_B2):
    bit_in_char = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
        3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
        3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
        2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
        3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
        5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
        2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
        4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
        3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
        4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
        5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
        5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8])

    B1 = _B1.astype('int')
    B2 = _B2.astype('int')

    n1 = B1.shape[0]
    n2, nwords = B2.shape

    Dh = np.zeros((n1, n2), 'uint16')
    for j in range(n1):
        for n in range(nwords):
            y = B1[j,n] ^ B2[:,n] # y is an array
            # TODO: Not sure which one is correct out of following two
            # Dh[j,:] = Dh[j,:] + bit_in_char[y+1]
            # Dh[j, :] = Dh[j, :] + bit_in_char[y]

            # following is correct if elements of B1 and B2 are 0s or 1s.
            # following line will not hold if B1 and B2 are created using compactbit() funtion.
            # for now assuming that B1 and B2 are NOT created using compactbit() function.
            Dh[j, :] = Dh[j,:] + y # this line is correct

    return Dh

'''
function cb = compactbit(b)

[nSamples nbits] = size(b);
nwords = ceil(nbits/8);
cb = zeros([nSamples nwords], 'uint8');

for j = 1:nbits
    w = ceil(j/8);
    cb(:,w) = bitset(cb(:,w), mod(j-1,8)+1, b(:,j));
end
'''
def compactbit(self, b):
    nSamples, nbits = b.shape
    nwords = ceil(nbits/8)
    cb = np.zeros(nSamples, nwords, dt='int')

    for j in range(nbits):
        w = ceil(j/8)
        cb[:, w] = bitset(cb[:, w], mod(j-1, 8)+1, b[:, j])

    return cb


