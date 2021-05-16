import numpy as np
from sklearn.decomposition import PCA
import json

def dumpToJson(dt, filename):
    """ dumps dict onto disk with given filename """
    with open(filename, 'w') as outfile:
        json.dump(dt, outfile)

dirpath_xv = '/mnt/f/mtp/fineTuning/fname_pred.npy'

print('reading finetuned vgg16 features...')
fname_pred = np.load(dirpath_xv, allow_pickle=True)[()]
fname_pred = {fname.split('/')[1] : pred for fname, pred in fname_pred.items()}

print('preparing training data...')
filenames = []
predictions = []
for fname, pred in fname_pred.items():
    filenames.append(fname)
    predictions.append(pred)
counter = len(predictions)
for i in range(counter):
    predictions.append(predictions[i])
predictions = np.array(predictions)

print(predictions.shape)
print('applying dimensionality reduction using PCA 25088D -> 4096D')
pca = PCA(n_components = 4096)
predictions = pca.fit_transform(predictions)

print(predictions.shape)
print(predictions[0, :40])

print('preparing and dumping the dict...')
fname_pred = { filenames[i] : predictions[i].tolist() for i in range(len(filenames)) }
dumpToJson(fname_pred, 'wiki_image_embedding.json')
