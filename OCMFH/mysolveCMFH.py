import numpy as np


def mySolveCMFH(self, X1, X2, lambda_, mu, gamma, numiter, bits):
    self.mu = mu
    self.gamma = gamma
    self.lambda_ = lambda_
    self.numiter = numiter
    self.bits = bits
    self.X1 = X1
    self.X2 = X2
    row, col = X1.shape()
    rowt = X2.shape[0]
    Y = np.random.randn(self.bits, col)
    P1 = np.random.randn(self.bits, row)
    P2 = np.random.randn(self.bits, rowt)
    U1 = np.random.randn(row, self.bits)
    U2 = np.random.randn(row, self.bits)
    threshold = 0.01
    lastF = 99999999
    iter = 1
    obj = np.array()

    while(True):
        # update U1 and U2
        U1 = self.X1 * Y.T / (Y * Y.T + self.gamma * np.eye(self.bits))
        U2 = self.X2 * Y.T / (Y * Y.T + self.gamma * np.eye(self.bits))

        # update Y
        Numerator = ((self.lambda_ * U1.T * U1) + ((1-self.lambda_) * U2.T * U2) + (2 * self.mu * np.eye(self.bits)) + (self.gamma * np.eye(self.bits)))
        Denominator = ((self.lambda_ * U1.T * X1) + ((1-self.lambda_) * U2.T * X2) + (self.mu * ( P1 * X1 + P2 * X2)))
        Y = Numerator / Denominator

        # Update P1 and P2
        P1 = Y * X1.T / (X1 * X1.T + self.gamma + np.eye(row))
        P2 = Y * X2.T / (X2 * X2.T + self.gamma + np.eye(rowt))

        # Compute object function
        norm1 = self.lambda_ * np.linalg.norm(X1 - U1 * Y, ord='fro')
        norm2 = (1 - self.lambda_) * np.linalg.norm(X2 - U2 * Y, ord='fro')
        norm3 = self.mu * np.linalg.norm(Y - P1 * X1, ord='fro')
        norm4 = self.mu * np.linalg.norm(Y - P2 * X2, ord='fro')
        norm5 = self.gamma * (np.linalg.norm(U1, ord='fro') + np.linalg.norm(U2, ord='fro') +np.linalg.norm(Y, ord='fro') + np.linalg.norm(P1, ord='fro') + np.linalg.norm(P2, ord='fro'))
        currentF = norm1 + norm2 + norm3 + norm4 + norm5
        # obj = [obj, currentF];
        obj.append(currentF)
        if lastF - currentF < threshold:
            break
        if iter >= self.numiter:
            break
        iter = iter + 1
        lastF = currentF

    return U1, U2, P1, P2, Y, obj



# obj is like an array which gets updated with currentF value everytime in loop
