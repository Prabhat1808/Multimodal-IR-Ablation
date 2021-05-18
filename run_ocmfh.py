from Framework.dataset import Dataset
from Framework.model import Parameters, Model
import numpy as np
from sklearn.decomposition import PCA
import random

xmedianet_filepath = '/mnt/f/mtp/dataset/dataset/xmedianet/'
pca = PCA(n_components = 128)

def chunkify(M, chunk_size):
    """
        Given a matrix M of m samples, make chunks of size
        chunk_size. Return list of chunks.
    """
    num_chunks = (M.shape[0] // chunk_size) + (M.shape[0] % chunk_size != 0)
    M_chunked = []
    for i in range(num_chunks):
        M_chunked.append(M[i*chunk_size:i*chunk_size + chunk_size, :])
    return M_chunked

def loader(dirpath, tag):
	""" reads the data from dirpath

		inputs:
		dirpath: path to the train data and labels
		tag: can be "train", "val", or "test"

		output:
		tag		data
		train	x_train : {'X1_train':X1_train, 'X2_train':X2_train}
				y_train : {'L_tr':L_tr}

		test	x_test : {'I_te':I_te, 'T_te':T_te}
				y_test : {'L_te':L_te}
	"""
	chunk_size = 5000
	identity = np.eye(200)
	print('Reading labels...')
	if (tag == 'test'):
		Y_test = []
		with open(dirpath + 'img_test_list.txt', 'r') as infile:
		    while (True):
		        line = infile.readline()
		        if (line == None) or (len(line.split()) != 2): break
		        Y_test.append(identity[int(line.split()[1])-1].tolist())
		Y_test = np.array(Y_test)
	if (tag == 'train'):
		with open(dirpath + 'img_train_list.txt', 'r') as infile:
		    Y_train = []
		    while (True):
		        line = infile.readline()
		        if (line == None) or (len(line.split()) != 2): break
		        Y_train.append(identity[int(line.split()[1])-1].tolist())
		Y_train = np.array(Y_train)
	    
	if (tag == 'test'): 
		print('preparing test features...')
		X1_test = np.loadtxt(dirpath + 'img_test_fea.txt') # 8K x 4096
		X2_test = np.loadtxt(dirpath + 'txt_test_fea.txt') # 8K x 300
		X1_test = pca.transform(X1_test)

	if (tag == 'train'):
		print('preparing train features...')
		img_train_fea = np.loadtxt(dirpath + 'img_train_fea.txt') # 32K x 4096
		text_train_fea = np.loadtxt(dirpath + 'txt_train_fea.txt') # 32K x 300

		print('Applying dimensionality reduction 4096D -> 128D...')
		img_train_fea = pca.fit_transform(img_train_fea)
		
		print('Shuffling the training data...')
		indices = [[i] for i in range(img_train_fea.shape[0])]
		random.shuffle(indices)
		indices = np.array(indices).flatten()
		img_train_fea = img_train_fea[indices, :]
		text_train_fea = text_train_fea[indices, :]
		Y_train = Y_train[indices, :]

		print('Chunkifying the training data...')
		X1_train = chunkify(img_train_fea, chunk_size)
		X2_train = chunkify(text_train_fea, chunk_size)

	if (tag == 'train'):
		return {'X1_train' : X1_train, 'X2_train':X2_train}, Y_train
	elif (tag == 'test'):
		return {'I_te' : X1_test.T, 'T_te' : X2_test.T}, Y_test

def dummyLoader(dirpath, tag):
	print('returning the dummy dataset...')
	X1_train = [ np.array([ [1.0, 1.1, 2.1], [2.0, 2, 1]]), np.array([[1.2, 2.2, 2.9], [9.1, 8.1, 0.1]]) ]
	X2_train = [ np.array([ [2.0, 6.1], [5.0, 6]]), np.array([[1.7, 1.2], [2.1, 2.1]]) ]
	I_te = np.array([[1.0, 3.4, 5.2], [2.3, 1.4, 5.8]]).T
	T_te = np.array([[1.4, 4.4], [2.5, 5.1]]).T
	L_te = np.array([[0, 1.0, 0.0], [1.0, 0.0, 0.0]])
	L_tr = np.array([[0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0, 0], [0, 1.0, 0]]) 
	if (tag == 'train'):
		return {'X1_train' : X1_train, 'X2_train':X2_train}, L_tr
	elif (tag == 'test'):
		return {'I_te' : I_te, 'T_te' : T_te}, L_te

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

def train(dataset_obj, params, hyperparams):
	""" Train parameteres on training data and returns 
		parameteres that will be used by the prediction function
		
		args:
			dataset_obj : Dataset() obj
			params : parameters() obj
			hyperparams : dict of hyperparameter values
	"""
	# preparing some parameters
	X1_train = dataset_obj.x_train['X1_train']
	X2_train = dataset_obj.x_train['X2_train']
	I_te = dataset_obj.x_test['I_te']
	T_te = dataset_obj.x_test['T_te']
	bits = hyperparams['bits']
	lambda_ = hyperparams['lambda_']
	mu = hyperparams['mu']
	gamma = hyperparams['gamma']
	iter = hyperparams['iter']
	cmfhiter = hyperparams['cmfhiter']

	# main training algo
	nstream = len(X1_train)     #TODO: 0 or 1 ?
	# Initialization ....check and add 
	Itrain = X1_train[0]
	Ttrain = X2_train[0]
	numdata = Itrain.shape[0]
	mean_I = np.mean(Itrain,axis=0) #1 ALgo 3
	mean_T = np.mean(Ttrain,axis=0)
	Itrain = Itrain - mean_I
	Ttrain = Ttrain - mean_T
	print("Batch: 1  Total: ", nstream)
	WI, WT, PI, PT, HH, obj = mysolveCMFH(Itrain.T, Ttrain.T, lambda_, mu, gamma, cmfhiter, bits) # 2 Algo 3

    # Train---2 to n

	mFea1 = Itrain.shape[1]
	mFea2 = Ttrain.shape[1]
	W1 = np.matmul(Itrain.T, HH.T)
	W2 = np.matmul(Ttrain.T, HH.T)
	H1 = np.matmul(HH, HH.T)
	H2 = H1.copy()
	F1 = np.matmul(HH,Itrain)
	F2 = np.matmul(HH,Ttrain)
	G1 = np.matmul(Itrain.T,Itrain) + gamma*np.eye(mFea1)
	G2 = np.matmul(Ttrain.T,Ttrain) + gamma*np.eye(mFea2)

	for i in range(1,nstream):
		print('stream: {} is running...'.format(i))
		# Initialization ....check and add
		Itrain = X1_train[i]
		Ttrain = X2_train[i]
		numdata_tmp = Itrain.shape[0]  #TODO: shouldn't it be shape[0]
		mean_Itmp = np.mean(Itrain,axis=0) # 4.1 Algo 3
		mean_Ttmp = np.mean(Ttrain,axis=0)
		mean_I = (1 / (numdata + numdata_tmp)) * np.add(numdata * mean_I, numdata_tmp*mean_Itmp)
		mean_T = (1 / (numdata + numdata_tmp)) * np.add(numdata * mean_T, numdata_tmp*mean_Ttmp)
		Itrain = np.subtract(Itrain,mean_I)
		Ttrain = np.subtract(Ttrain, mean_T)
		numdata = numdata + numdata_tmp
		#  4.3 Algo3
		WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj =  mysolveOCMFH(Itrain.T, Ttrain.T, WI, WT, PI, PT, W1, W2, H1, H2, F1, F2, G1, G2, HH, obj, lambda_, mu, gamma, iter)
         
	Y_tr =np.sign(np.subtract( HH, np.mean(HH, axis = 1).reshape((-1, 1))).T)  # returns sign of each element
	Y_tr[Y_tr<0] = 0

	#B_Tr = compactbit(Y_tr) #TODO: use it if able to pythonize compactbit function
	B_Tr = Y_tr.copy()
	B_Ir = B_Tr.copy()
	return {'PI':PI, 'PT':PT, 'HH':HH, 'B_Tr' : B_Tr, 'B_Ir' : B_Ir, 'B_Te' : None, 'B_Ie' : None}, obj, None

def hammingdist(_B1,_B2):
    B1 = _B1.astype('int')
    B2 = _B2.astype('int')

    n1 = B1.shape[0]
    n2, nwords = B2.shape

    Dh = np.zeros((n1, n2), 'uint16')
    for j in range(n1):
        for n in range(nwords):
            y = B1[j,n] ^ B2[:,n] # y is an array
            # following is correct if elements of B1 and B2 are 0s or 1s.
            # following line will not hold if B1 and B2 are created using compactbit() funtion.
            # for now assuming that B1 and B2 are NOT created using compactbit() function.
            Dh[j, :] = Dh[j,:] + y # this line is correct

    return Dh

def predict(dataset_obj, params, tag):
	"""Given parameters learnt and necessary for prediction, 
		this function predicts hash codes of test dataset

	arg:
		dataset_obj : Dataset() obj that contains train and test data
		params : Parameters() consisting of imp weights
		tag : One of the following: 'train', 'test', or 'val'

	output:
		n_samples: #samples predicted
		results: matrix query_size x train_size
				 It is ranked list of items retrieved from train sample.
				 There should be two matrices, one for ItoT and second for
				 TtoI. 
		logs: 
	"""
	# Fetching some params
	PI = params['PI']
	PT = params['PT']
	HH = params['HH']
	I_te = dataset_obj.x_test['I_te']
	T_te = dataset_obj.x_test['T_te']
	# Testing
	Yi_te  = np.sign(np.subtract(np.matmul(PI, I_te), np.mean(HH, axis=1).reshape((-1, 1)))).T  # returns sign
	Yt_te  = np.sign(np.subtract(np.matmul(PT, T_te), np.mean(HH, axis=1).reshape((-1, 1)))).T  # returns sign
	Yi_te[Yi_te < 0] = 0
	Yt_te[Yt_te < 0] = 0

	B_Te = Yt_te.copy()
	B_Ie = Yi_te.copy()
	B_Ir = params['B_Ir']
	B_Tr = params['B_Tr']

	itot_hamming_dist = hammingdist(B_Ie, B_Tr)
	ttoi_hamming_dist = hammingdist(B_Te, B_Ir)
	itot_ranked_results = np.argsort(itot_hamming_dist, axis = 1)
	ttoi_ranked_results = np.argsort(ttoi_hamming_dist, axis = 1)

	results = {'itot_ranked_results':itot_ranked_results, 'ttoi_ranked_results':ttoi_ranked_results}

	return I_te.shape[1], results, None

data = Dataset((xmedianet_filepath, xmedianet_filepath, xmedianet_filepath), 
				loader, read_directories=(True, False, True))
data.load_data()
hyperparams = {'bits':32, 'lambda_':0.5, 'mu':100, 'gamma':0.001, 'iter':100, 'cmfhiter':100}
params = Parameters({'PI':None, 'PT':None, 'HH':None, 'B_Tr' : None, 'B_Ir' : None, 'B_Te' : None, 'B_Ie' : None})
model = Model(	train, 
				hyperparams, 
				data, 
				params, 
				None, #params_verification 
				predict, #prediction_function
				None) #evaluation_metrics
model.train_model()
model.predict('test')
model.evaluate(data.get_train_labels(), data.get_test_labels())