import cPickle
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

from Framework.dataset import Dataset
from Framework.model import Model
from ACMR.adv_crossmodal_simple_nuswide import AdvCrossModalSimple, ModelParams, DataIter


def dummy_predict(dataset_obj, params, tag):
    n_samples = params['number_of_queries']
    results = {'itot_ranked_results': params['itot_ranked_results'].copy(),
               'ttoi_ranked_results': params['ttoi_ranked_results'].copy()}
    return n_samples, results, None


def data_loader(dirpath, tag):
    if tag == 'train':
        with open('./data/wikipedia_dataset/train_img_feats.pkl', 'rb') as f:
            train_img_feats = cPickle.load(f)
        with open('./data/wikipedia_dataset/train_txt_vecs.pkl', 'rb') as f:
            train_txt_vecs = cPickle.load(f)
        with open('./data/wikipedia_dataset/train_labels.pkl', 'rb') as f:
            train_labels = cPickle.load(f)

        return {'img_train': train_img_feats, 'txt_train': train_txt_vecs}, {'train_labels': train_labels}

    if tag == 'test':
        with open('./data/wikipedia_dataset/test_img_feats.pkl', 'rb') as f:
            test_img_feats = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_txt_vecs.pkl', 'rb') as f:
            test_txt_vecs = cPickle.load(f)
        with open('./data/wikipedia_dataset/test_labels.pkl', 'rb') as f:
            test_labels = cPickle.load(f)
        return {'img_test': test_img_feats, 'txt_test': test_txt_vecs}, {'test_labels': test_labels}


def summarizeDataset(X, Y, tag):
    if tag == 'train':
        return {'num_samples': Y['train_labels'].shape[0],
                'img_feature_dim': X['img_train'][0].shape[1],
                'text_feature_dim': X['txt_train'][0].shape[1],
                'num_classes': Y['train_labels'].shape[1]}
    if tag == 'test':
        return {'num_samples': Y['test_labels'].shape[0],
                'img_feature_dim': X['img_test'][0].shape[1],
                'text_feature_dim': X['txt_test'][0].shape[1],
                'num_classes': Y.shape[1]}


def predict_(dataset_obj, params, tag, sess):
    if tag == 'test':
        data_iter = params['dataIter']
        emb_v = params['emb_v']
        emb_w = params['emb_w']
        # sess = params['sess']
        visual_feats = params['visual_feats']
        word_vecs = params['word_vecs']
        test_img_feats_trans = []
        test_txt_vecs_trans = []
        test_labels = []
        for feats, vecs, labels, i in data_iter.test_data():
            feats_trans = sess.run(emb_v, feed_dict={visual_feats: feats})
            vecs_trans = sess.run(emb_w, feed_dict={word_vecs: vecs})
            # print("{0}".format(np.shape(labels)))
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
    retrieval = {'itot_ranked_results': np.array(retrieval_i_t), 'ttoi_ranked_results': np.array(retrieval_t_i),
                 'number_of_queries': number_of_queries}
    return number_of_queries, retrieval, None


def train(dataset_obj, parameters, hyperparams):
    print("Inside Model -- train ( basically main of ACMR)")
    graph = tf.Graph()
    model_params = hyperparams
    model_params.update()
    with graph.as_default():
        dataIter = DataIter(hyperparams.batch_size, dataset_obj)
        model = AdvCrossModalSimple(model_params, dataIter)
    with tf.Session(graph=graph) as sess:
        parameters_, losses, logs = model.train(sess)
        # model.eval(sess)
        number_of_queries, retrieval, logs = predict_(dataset_obj, parameters_, 'test', sess)
        return retrieval, losses, logs


def main(_):
    nuswide_filepath_train = "./data/wikipedia_dataset/"
    nuswide_filepath_valid = "."
    nuswide_filepath_test = "./data/wikipedia_dataset/"
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
    y_mat_label = np.array(mat_label)
    data = y_mat_label[0:640]
    encoded = to_categorical(data)
    model.evaluate(encoded, encoded)
    model.save_stats("acmr_wikipedia_stats.npy")


if __name__ == '__main__':
    tf.app.run()
