import numpy as np

# bits looks like representing the binary code length
def mysolveCMFH(X1, X2, lambda_, mu, gamma, numiter, bits):
    row, col = X1.shape
    rowt = X2.shape[0]
    Y = np.random.randn(bits, col)
    P1 = np.random.randn(bits, row)
    P2 = np.random.randn(bits, rowt)
    U1 = np.random.randn(row, bits)
    U2 = np.random.randn(rowt, bits)     #TODO: It must be rowt here.
    threshold = 0.01
    lastF = 99999999
    iter = 1
    obj = [] # stores loss value for each training iteration

    while(True):
        # update U1 and U2
        U1 = np.matmul(np.matmul(X1, Y.T), np.linalg.inv( np.matmul(Y, Y.T) + (gamma/lambda_) * np.eye(bits))) # line 3.1, algo-1
        U2 = np.matmul(np.matmul(X2, Y.T), np.linalg.inv( np.matmul(Y, Y.T) + (gamma/(1-lambda_)) * np.eye(bits))) # line 3.1, algo-1

        # update Y
        Numerator = np.linalg.inv((lambda_ * np.matmul(U1.T, U1)) + ((1-lambda_) * np.matmul(U2.T, U2)) + ((2 * mu + gamma) * np.eye(bits)))
        Denominator = ((lambda_ * np.matmul(U1.T, X1)) + ((1-lambda_) * np.matmul(U2.T, X2)) + (mu * ( np.matmul(P1, X1) + np.matmul(P2, X2))))
        Y = np.matmul(Numerator, Denominator) #line 3.3, algo-1

        # Update P1 and P2
        P1 = np.matmul(np.matmul(Y, X1.T), np.linalg.inv( np.matmul(X1, X1.T) + (gamma/mu) * np.eye(row)))
        P2 = np.matmul(np.matmul(Y, X2.T), np.linalg.inv( np.matmul(X2, X2.T) + (gamma/mu) * np.eye(rowt)))

        # Compute object function
        norm1 = lambda_ * np.linalg.norm(X1 - np.matmul(U1, Y), ord='fro')
        norm2 = (1 - lambda_) * np.linalg.norm(X2 - np.matmul(U2, Y), ord='fro')
        norm3 = mu * np.linalg.norm(Y - np.matmul(P1, X1), ord='fro')
        norm4 = mu * np.linalg.norm(Y - np.matmul(P2, X2), ord='fro')
        norm5 = gamma * (np.linalg.norm(U1, ord='fro') + np.linalg.norm(U2, ord='fro') +\
                np.linalg.norm(Y, ord='fro') + np.linalg.norm(P1, ord='fro') + np.linalg.norm(P2, ord='fro'))
        currentF = norm1**2 + norm2**2 + norm3**2 + norm4**2 + norm5**2
        obj.append(currentF)
        print('{}th iteration of CMFH with loss: {}'.format(iter, currentF))
        if lastF - currentF < threshold:
            break
        if iter >= numiter:
            break
        iter = iter + 1
        lastF = currentF
    
    return U1, U2, P1, P2, Y, obj



# obj is like an array which gets updated with currentF value everytime in loop
