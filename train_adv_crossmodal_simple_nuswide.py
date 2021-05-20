import tensorflow as tf
from models.adv_crossmodal_simple_nuswide import AdvCrossModalSimple, ModelParams, DataIter
from Framework.dataset import Dataset
from Framework.model import Parameters, Model

import pickle
import numpy as np


def data_loader(dirpath, tag):
    if tag == 'train':
        with open(dirpath + 'img_train_id_feats.pkl') as f:
            train_img_feats = pickle.load(f)
        with open(dirpath + 'train_id_bow.pkl', 'rb') as f:
            train_txt_vecs = pickle.load(f)
        with open(dirpath + 'train_id_label_map.pkl', 'rb') as f:
            train_labels = pickle.load(f)
        with open(dirpath + 'train_ids.pkl', 'rb') as f:
            train_ids = pickle.load(f)
        with open(dirpath + 'train_id_label_single.pkl', 'rb') as f:
            train_labels_single = pickle.load(f)
        return {'img_train': train_img_feats, 'txt_train': train_txt_vecs}, {'train_labels': train_labels,
                                                                             'train_labels_single': train_labels_single,
                                                                             'train_ids': train_ids}

    if tag == 'test':
        with open(dirpath + 'img_test_id_feats.pkl', 'rb') as f:
            test_img_feats = pickle.load(f)
        with open(dirpath + 'test_id_bow.pkl', 'rb') as f:
            test_txt_vecs = pickle.load(f)
        with open(dirpath + 'test_id_label_map.pkl', 'rb') as f:
            test_labels = pickle.load(f)
        with open(dirpath + 'test_ids.pkl', 'rb') as f:
            test_ids = pickle.load(f)
        with open(dirpath + 'test_id_label_single.pkl', 'rb') as f:
            test_labels_single = pickle.load(f)
        return {'img_test': test_img_feats, 'txt_test': test_txt_vecs}, {'test_labels': test_labels,
                                                                         'test_labels_single': test_labels_single,
                                                                         'test_ids': test_ids}

def predict_(dataset_obj, params, tag,sess):
  if tag =='test':
    data_iter = params['dataIter']
    emb_v = params['emb_v']
    emb_w = params['emb_w']
    # sess = params['sess']
    visual_feats = params['visual_feats']
    word_vecs = params['word_vecs']
    test_img_feats_trans = []
    test_txt_vecs_trans = []
    test_labels = []
    for feats, vecs, _, labels, i in data_iter.test_data():
        feats_trans = sess.run(emb_v, feed_dict={visual_feats: feats})
        vecs_trans = sess.run(emb_w, feed_dict={word_vecs: vecs})
        #print("{0}".format(np.shape(labels)))
        test_labels += list(labels)
        for ii in range(len(feats)):
            test_img_feats_trans.append(feats_trans[ii])
            test_txt_vecs_trans.append(vecs_trans[ii])
    test_img_feats_trans = np.asarray(test_img_feats_trans)
    test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)

    retrieval_t_i = []
    number_of_queries = len(test_txt_vecs_trans)
    for i in range(len(test_txt_vecs_trans)):
        query_label = test_labels[i]
        # distances and sort by distances
        wv = test_txt_vecs_trans[i]
        diffs = test_img_feats_trans - wv
        dists = np.linalg.norm(diffs, axis=1)
        sorted_idx = np.argsort(dists)
        retrieval_t_i.append(sorted_idx)

    retrieval_i_t = []
    for i in range(len(test_img_feats_trans)):
        query_img_feat = test_img_feats_trans[i]
        ground_truth_label = test_labels[i]

        # calculate distance and sort
        diffs = test_txt_vecs_trans - query_img_feat
        dists = np.linalg.norm(diffs, axis=1)
        sorted_idx = np.argsort(dists)
        retrieval_i_t.append(sorted_idx)
    retrieval = {'itot_ranked_results':np.array(retrieval_i_t), 'ttoi_ranked_results':np.array(retrieval_t_i), 'number_of_queries': number_of_queries }
  return number_of_queries, retrieval, None

def dummy_predict(dataset_obj, params, tag):
  n_samples = params['number_of_queries']
  results = {'itot_ranked_results':params['itot_ranked_results'].copy(), 'ttoi_ranked_results':params['ttoi_ranked_results'].copy()}
  return n_samples, results, None

def train(dataset_obj, parameters, hyperparams):
    print("Inside Model -- train ( basically main of ACMR)")
    graph = tf.Graph()
    model_params = hyperparams
    model_params.update()
    with graph.as_default():
        dataIter = DataIter(hyperparams.batch_size, dataset_obj)
        model = AdvCrossModalSimple(model_params, dataIter)
    with tf.Session(graph=graph) as sess:
        parameters_, losses, logs = model.train_(sess)
        model.eval(sess)
        number_of_queries, retrieval, logs = predict_(dataset_obj, parameters_, 'test', sess)
        return retrieval, losses, logs


def main(_):
    nuswide_filepath_train = "./models/data/nuswide_train/"
    nuswide_filepath_valid = "."
    nuswide_filepath_test = "./models/data/nuswide_test/"
    data = Dataset((nuswide_filepath_train, nuswide_filepath_valid, nuswide_filepath_test), data_loader,
                   read_directories=(True, False, True))
    data.load_data()
    print("Data loaded successfully")
    hyperparams = ModelParams()
    model = Model(train, hyperparams, data, prediction_function=dummy_predict)
    model.train_model()
    model.predict('test')
    label_ = data.y_test['test_labels']
    mat_label = []
    for key in label_:
        mat_label.append(label_[key])
    y_mat_label = np.array(mat_label)[0:1984, :]
    # number of samples * number of class
    model.evaluate(y_mat_label, y_mat_label)
    model.save_stats("nuswide_acmr_stats.npy")
    print("here")


if __name__ == '__main__':
    tf.app.run()