from __future__ import print_function
from Framework.dataset import Dataset
from Framework.model import Parameters, Model
# from sklearn.decomposition import PCA
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from random import shuffle
from ACMR.flip_gradient import flip_gradient

nuswide_filepath = './nuswide'
wikipedia_dataset_filepath = './wikipedia_dataset'


# pca = PCA(n_components = 128)
def main(_):
    graph = tf.Graph()
    data = Dataset((nuswide_filepath, nuswide_filepath, nuswide_filepath), DataIter,
                   read_directories=(True, False, True))
    data.load_data()
    hyperparams = {}
    params = Parameters(
        {'batch_size': 64, 'visual_feat_dim': 4096, 'word_vec_dim': 1000, 'lr_emb': 0.0001, 'lr_domain': 0.0001,
         'top_k': 50,
         'semantic_emb_dim': 40, 'dataset_name': 'nuswide', 'model_name': 'adv_semantic_zsl',
         'model_dir': 'adv_semantic_zsl_%d_%d_%d' % (4096, 1000, 40)})
    with graph.as_default():
        model = Model(train,
                      hyperparams,
                      data,
                      params,
                      None,  # params_verification
                      predict,  # prediction_function
                      None)  # evaluation_metrics
    with tf.Session(graph=graph) as sess:
        model.train_model(sess)
        model.predict('test', sess)
        model.evaluate(data.get_train_labels(), data.get_test_labels(), sess)


# def readtxt(filename):
#     all_filename = []
#     f = open(filename)
#     line = f.readline()
#     while line:
#         line_ = line.strip('\n')
#         all_filename.append(line_)
#         line = f.readline()
#     f.close()
#     return all_filename


def DataIter(dirpath, tag):
    with open(dirpath + 'img_train_id_feats.pkl', 'rb') as f:
        train_img_feats = pickle.load(f)
    with open(dirpath + 'train_id_bow.pkl', 'rb') as f:
        train_txt_vecs = pickle.load(f)
    with open(dirpath + 'train_id_label_map.pkl', 'rb') as f:
        train_labels = pickle.load(f)
    with open(dirpath + 'img_test_id_feats.pkl', 'rb') as f:
        test_img_feats = pickle.load(f)
    with open(dirpath + 'test_id_bow.pkl', 'rb') as f:
        test_txt_vecs = pickle.load(f)
    with open(dirpath + 'test_id_label_map.pkl', 'rb') as f:
        test_labels = pickle.load(f)
    with open(dirpath + 'train_ids.pkl', 'rb') as f:
        train_ids = pickle.load(f)
    with open(dirpath + 'test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    with open(dirpath + 'train_id_label_single.pkl', 'rb') as f:
        train_labels_single = pickle.load(f)
    with open(dirpath + 'test_id_label_single.pkl', 'rb') as f:
        test_labels_single = pickle.load(f)

    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)
    batch_size = 64
    num_train_batch = len(train_ids) / batch_size
    num_test_batch = len(test_ids) / batch_size

    # For training data
    if tag == 'train':
        for i in range(int(num_train_batch)):
            batch_img_ids = train_ids[i * batch_size: (i + 1) * batch_size]
            batch_img_feats = [train_img_feats[n] for n in batch_img_ids]
            batch_txt_vecs = [train_txt_vecs[n] for n in batch_img_ids]
            batch_labels = [train_labels[n] for n in batch_img_ids]
            batch_labels_single = np.array([train_labels_single[n] for n in batch_img_ids])
            return {'img_train': batch_img_feats, 'txt_train': batch_txt_vecs}, {'train_labels': batch_labels,
                                                                                 'train_labels_single': batch_labels_single,
                                                                                 'train_ids': i}

    if tag == 'test':
        for i in range(int(num_test_batch)):
            batch_img_ids = test_ids[i * batch_size: (i + 1) * batch_size]
            batch_img_feats = [test_img_feats[n] for n in batch_img_ids]
            batch_txt_vecs = [test_txt_vecs[n] for n in batch_img_ids]
            batch_labels = [test_labels[n] for n in batch_img_ids]
            batch_labels_single = [test_labels_single[n] for n in batch_img_ids]
            return {'img_test': batch_img_feats, 'txt_test': batch_txt_vecs}, {'test_labels': batch_labels,
                                                                               'test_labels_single': batch_labels_single,
                                                                               'test_ids': i}


def visual_feature_embed(X, params, is_training=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = tf.nn.tanh(slim.fully_connected(X, 4096, scope='vf_fc_0'))
        net = tf.nn.tanh(slim.fully_connected(net, 1000, scope='vf_fc_1'))
        net = tf.nn.tanh(slim.fully_connected(net, params['semantic_emb_dim'], scope='vf_fc_2'))
    return net


def label_embed(L, params, is_training=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = tf.nn.tanh(slim.fully_connected(L, 1000, scope='le_fc_0'))
        net = tf.nn.tanh(slim.fully_connected(net, 300, scope='le_fc_1'))
        net = tf.nn.tanh(slim.fully_connected(net, params['semantic_emb_dim'], scope='le_fc_2'))
    return net


def label_classifier(X, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = slim.fully_connected(X, 10, scope='lc_fc_0')
    return net


def domain_classifier(E, l, params, is_training=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        E = flip_gradient(E, l)
        net = slim.fully_connected(E, params['semantic_emb_dim'] / 2, scope='dc_fc_0')
        net = slim.fully_connected(net, params['semantic_emb_dim'] / 4, scope='dc_fc_1')
        net = slim.fully_connected(net, 2, scope='dc_fc_2')
    return net


def train(dataset_obj, params, hyperparams, sess):
    visual_feats = tf.placeholder(tf.float32, [None, params['visual_feat_dim']])
    word_vecs = tf.placeholder(tf.float32, [None, params['word_vec_dim']])
    y = tf.placeholder(tf.int32, [params['batch_size'], 10])
    y_single = tf.placeholder(tf.int32, [params['batch_size'], 1])
    l = tf.placeholder(tf.float32, [])
    emb_v = visual_feature_embed(visual_feats)
    emb_w = label_embed(word_vecs)

    emb_v_ = tf.reduce_sum(emb_v, axis=1, keep_dims=True)
    emb_w_ = tf.reduce_sum(emb_w, axis=1, keep_dims=True)
    distance_map = tf.matmul(emb_v_, tf.ones([1, params['batch_size']])) - tf.matmul(emb_v,
                                                                                     tf.transpose(
                                                                                         emb_w)) + \
                   tf.matmul(tf.ones([params['batch_size'], 1]), tf.transpose(emb_w_))
    mask_initial = tf.to_float(
        tf.matmul(y_single, tf.ones([1, params['batch_size']], dtype=tf.int32)) - \
        tf.matmul(tf.ones([params['batch_size'], 1], dtype=tf.int32), tf.transpose(y_single)))
    mask = tf.to_float(tf.not_equal(mask_initial, tf.zeros_like(mask_initial)))
    masked_dissimilar_loss = tf.multiply(distance_map, mask)
    dissimilar_loss = tf.reduce_mean(tf.maximum(0., 0.1 * tf.ones_like(mask) - masked_dissimilar_loss))
    similar_loss = tf.sqrt(2 * tf.nn.l2_loss(emb_v - emb_w))
    similar_loss = tf.reduce_mean(similar_loss)
    logits_v = label_classifier(emb_v)
    logits_w = label_classifier(emb_w, reuse=True)
    label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_v) + \
                 tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_w)
    label_loss = tf.reduce_mean(label_loss)
    emb_loss = 50 * label_loss + similar_loss + 0.2 * dissimilar_loss
    emb_v_class = domain_classifier(emb_v, l)
    emb_w_class = domain_classifier(emb_w, l, reuse=True)

    all_emb_v = tf.concat([tf.ones([params['batch_size'], 1]),
                           tf.zeros([params['batch_size'], 1])], 1)
    all_emb_w = tf.concat([tf.zeros([params['batch_size'], 1]),
                           tf.ones([params['batch_size'], 1])], 1)
    domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=emb_v_class, labels=all_emb_w) + \
                        tf.nn.softmax_cross_entropy_with_logits(logits=emb_w_class, labels=all_emb_v)
    domain_class_loss = tf.reduce_mean(domain_class_loss)
    t_vars = tf.trainable_variables()
    vf_vars = [v for v in t_vars if 'vf_' in v.name]
    le_vars = [v for v in t_vars if 'le_' in v.name]
    dc_vars = [v for v in t_vars if 'dc_' in v.name]
    lc_vars = [v for v in t_vars if 'lc_' in v.name]

    emb_train_op = tf.train.AdamOptimizer(
        learning_rate=params['lr_emb'],
        beta1=0.5).minimize(emb_loss, var_list=le_vars + vf_vars)
    domain_train_op = tf.train.AdamOptimizer(
        learning_rate=params['lr_domain'],
        beta1=0.5).minimize(domain_class_loss, var_list=dc_vars)

    tf.initialize_all_variables().run()
    saver = tf.train.Saver()  # ------------------
    params['epoch'] = 50
    for epoch in range(params['epoch']):

        p = float(epoch) / params['epoch']
        l = 2. / (1. + np.exp(-10. * p)) - 1
        for batch_feat, batch_vec, batch_labels, batch_labels_single, idx in self.data_iter.train_data():
            # sess.run([total_train_op], feed_dict={visual_feats: batch_feat, word_vecs: batch_vec, y: b,l: l})
            sess.run([emb_train_op, domain_train_op],
                     feed_dict={
                         visual_feats: batch_feat,
                         word_vecs: batch_vec,
                         y: batch_labels,
                         y_single: batch_labels_single[:, np.newaxis],
                         l: l})

            label_loss_val, similar_loss_val, emb_loss_val, domain_loss_val, dissimilar_loss_val = sess.run(
                [label_loss, similar_loss, emb_loss, domain_class_loss, dissimilar_loss],
                feed_dict={visual_feats: batch_feat,
                           word_vecs: batch_vec,
                           y: batch_labels,
                           y_single: batch_labels_single[:, np.newaxis],
                           l: l})
            print(
                'Epoch: [%2d][%4d/%4d] time: %4.4f, emb_loss: %.8f, domain_loss: %.8f, label_loss: %.8f, similar_loss: %.8f, suaimilar_loss: %.8f' % (
                    epoch, idx, self.data_iter.num_train_batch, 0, emb_loss_val,
                    domain_loss_val, label_loss_val, similar_loss_val, dissimilar_loss_val
                ))


def predict(dataset_obj, params, tag):
    # do this while loading data
    with open('./data/nuswide/test_label_map.pkl', 'rb') as fpkl:
        test_labels = pickle.load(fpkl)

    k = params['top_k']
    avg_precs = []
    for i in range(len(test_labels)):
        query_label = test_labels[i]

        # distances and sort by distances
        sorted_idx = range(len(test_labels))
        shuffle(sorted_idx)

        # for each k do top-k
        precs = []
        for topk in range(1, k + 1):
            hits = 0
            top_k = sorted_idx[0: topk]
            if query_label != test_labels[top_k[-1]]:
                continue
            for ii in top_k:
                retrieved_label = test_labels[ii]
                if query_label != retrieved_label:
                    hits += 1
            precs.append(float(hits) / float(topk))
        avg_precs.append(np.sum(precs) / float(k))
    mean_avg_prec = np.mean(avg_precs)
    print('[Eval - random] mAP: %f in %4.4fs' % (mean_avg_prec))


if __name__ == '__main__':
    tf.app.run()
