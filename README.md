# Multimodal-IR-Ablation

### Dataset: XMediaNet
#### Available to download at [link](https://drive.google.com/file/d/1OYv7IK5OdDmBQ_pRXMugnqZzR_j7HOsu/view?usp=sharing)

## Parameters
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

	- X1_train : c * d1 * n 	( training data for modality 1)
	- X2_train : c * d2 * n 	( training data for modality 2)
	- Y_train  : n * p		( training labels shared b/w modality 1, 2)
	- X1_test  : m * d1		( testing data for modality 1)
	- X2_test  : m * d2		( testing data for modality 2)
	- Y_test   : m * p		( testing labels shared b/w modality 1, 2)

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
	- B1_Q : L * m			( Hash codes for test samples from modality 1)
	- B2_Q : L * m			( Hash codes for test samples from modality 2)

## Evaluation stage
	Yet to define

## Variable notation for original paper vs Implementation
V<sup>t</sup> :    H  
U<sup>t</sup> :   WI/WT  
X<sup>t</sup> :   Itrain/Ttrain  
P<sup>t</sup> :   PI/PT  
F<sup>t</sup> :   F1/F2  
W<sup>t</sup> :   G1/G2  
E<sup>t</sup> :   W1/W2  
C<sup>t</sup> :   H1/H2  
V<sup>t-1</sup><sub>new</sub> :   HH  
U<sup>t</sup> :   U<sub>new</sub>  
U<sup>t-1</sup> :   U<sub>old</sub>  
Loss per iteration :   obj  

## main function parameters
	- B_I : Image Training --> B_Ir
	- B_T : Text Training --> B_Tr
	- tB_I : Image Testing --> B_Ie
	- tB_T : Text Testing --> B_Te


