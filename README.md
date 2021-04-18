# Multimodal-IR-Ablation

## Parameters(just for illustration purpose)
	- n : number of training samples
	- m : number of testing samples
	- p : number of ground truth categories
	- c : n training samples are divided into c chunks. Hence n/c samples per chunk.
	- d1: feature dimension of items from modality 1
	- d2: feature dimension of items from modality 2
	- L : hash code length

## Training stage

### Input matrices
	As of now only the training features are in chunk.

	- X1\_train : c * d1 * n 	( training data for modality 1)
	- X2\_train : c * d2 * n 	( training data for modality 2)
	- Y\_train  : n * p		( training labels shared b/w modality 1, 2)
	- X1\_test  : m * d1		( testing data for modality 1)
	- X2\_test  : m * d2		( testing data for modality 2)
	- Y\_test   : m * p		( testing labels shared b/w modality 1, 2)

### Output matrices
	- B  : L * n			( binary hash codes for training data)
	- P1 : L * d1			( projection matrix for modality 1)
	- P2 : L * d2			( projection matrix for modality 2)
	- u1 : d1 * 1			( mean vector for modality 1, stores mean per dimension)
	- u2 : d2 * 1			( mean vector for modality 2, stores mean per dimension)

## Inference stage

### Input matrices
	- P1 : L * d1			( projection matrix for modality 1)
	- P2 : L * d2			( projection matrix for modality 2)
	- u1 : d1 * 1			( mean vector for modality 1, stores mean per dimension)
	- u2 : d2 * 1			( mean vector for modality 2, stores mean per dimension)

### Output matricies
	- B1\_Q : L * m			( Hash codes for test samples from modality 1)
	- B2\_Q : L * m			( Hash codes for test samples from modality 2)

## Evaluation stage
	Yet to define
