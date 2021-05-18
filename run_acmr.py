from __future__ import print_function
from Framework.dataset import Dataset
from Framework.model import Parameters, Model
# from sklearn.decomposition import PCA
from models.adv_crossmodal_simple_nuswide import AdvCrossModalSimple, ModelParams
# from models.base_model import BaseModel, BaseModelParams, BaseDataIter
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
# from random import shuffle
# import sklearn.preprocessing
import os, time
from models.flip_gradient import flip_gradient

nuswide_filepath = './nuswide'
wikipedia_dataset_filepath = './wikipedia_dataset'
# pca = PCA(n_components = 128)
def main(_):
    graph = tf.Graph()

    with graph.as_default():
        data = Dataset((nuswide_filepath, nuswide_filepath, nuswide_filepath), DataIter,
                       read_directories=(True, False, True))
        data.load_data()
        hyperparams = {}
        params = Parameters(
            {'batch_size': 64, 'visual_feat_dim': 4096, 'word_vec_dim': 1000, 'lr_emb': 0.0001, 'lr_domain': 0.0001,
             'top_k': 50,
             'semantic_emb_dim': 40, 'dataset_name': 'nuswide', 'model_name': 'adv_semantic_zsl',
             'model_dir': 'adv_semantic_zsl_%d_%d_%d' % (4096, 1000, 40)})
        # 'checkpoint_dir': 'checkpoint', 'sample_dir': 'samples', 'dataset_dir': 'samples', 'dataset_dir': '.', 'log_dir': 'logs' })
        model = Model(train,
                      hyperparams,
                      data,
                      params,
                      None,  # params_verification
                      predict,  # prediction_function
                      None)  # evaluation_metrics
        model.train_model()
        model.predict('test')
        model.evaluate(data.get_train_labels(), data.get_test_labels())

    with tf.Session(graph=graph) as sess:
        model.train(sess)
        # model.eval_random_rank()
        model.eval(sess)


def readtxt(filename):
    all_filename = []
    f = open(filename)
    line = f.readline()
    while line:
        line_ = line.strip('\n')
        all_filename.append(line_)
        line = f.readline()
    f.close()
    return all_filename

# def chunkify(M, chunk_size):
#     """
#         Given a matrix M of m samples, make chunks of size
#         chunk_size. Return list of chunks.
#     """
#     num_chunks = (M.shape[0] // chunk_size) + (M.shape[0] % chunk_size != 0)
#     M_chunked = []
#     for i in range(num_chunks):
#         M_chunked.append(M[i * chunk_size:i * chunk_size + chunk_size, :])
#     return M_chunked

def DataIter(dirpath, tag):
    with open(dirpath+'img_train_id_feats.pkl', 'rb') as f:
        train_img_feats = pickle.load(f)
    with open(dirpath+'train_id_bow.pkl', 'rb') as f:
        train_txt_vecs = pickle.load(f)
    with open(dirpath+'train_id_label_map.pkl', 'rb') as f:
        train_labels = pickle.load(f)
    with open(dirpath+'img_test_id_feats.pkl', 'rb') as f:
        test_img_feats = pickle.load(f)
    with open(dirpath+'test_id_bow.pkl', 'rb') as f:
        test_txt_vecs = pickle.load(f)
    with open(dirpath+'test_id_label_map.pkl', 'rb') as f:
        test_labels = pickle.load(f)
    with open(dirpath+'train_ids.pkl', 'rb') as f:
        train_ids = pickle.load(f)
    with open(dirpath+'test_ids.pkl', 'rb') as f:
        test_ids = pickle.load(f)
    with open(dirpath+'train_id_label_single.pkl', 'rb') as f:
        train_labels_single = pickle.load(f)
    with open(dirpath+'test_id_label_single.pkl', 'rb') as f:
        test_labels_single = pickle.load(f)

    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)
    batch_size=64
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
            return {'img_train':batch_img_feats, 'txt_train':batch_txt_vecs}, {'train_labels':batch_labels, 'train_labels_single':batch_labels_single,'train_ids': i}

    if tag == 'test':
        for i in range(int(num_test_batch)):
            batch_img_ids = test_ids[i*batch_size : (i+1)*batch_size]
            batch_img_feats = [test_img_feats[n] for n in batch_img_ids]
            batch_txt_vecs = [test_txt_vecs[n] for n in batch_img_ids]
            batch_labels = [test_labels[n] for n in batch_img_ids]
            batch_labels_single = [test_labels_single[n] for n in batch_img_ids]
            return {'img_test':batch_img_feats, 'txt_test':batch_txt_vecs}, {'test_labels':batch_labels, 'test_labels_single':batch_labels_single, 'test_ids': i}


class AdvCrossModalSimple(BaseModel):
    def __init__(self, model_params):
        BaseModel.__init__(self, model_params)
        self.data_iter = DataIter(self.model_params.batch_size)
        self.visual_feats = tf.placeholder(tf.float32, [None, self.model_params.visual_feat_dim])
        self.word_vecs = tf.placeholder(tf.float32, [None, self.model_params.word_vec_dim])
        self.y = tf.placeholder(tf.int32, [self.model_params.batch_size, 10])
        self.y_single = tf.placeholder(tf.int32, [self.model_params.batch_size, 1])
        self.l = tf.placeholder(tf.float32, [])
        self.emb_v = self.visual_feature_embed(self.visual_feats)
        self.emb_w = self.label_embed(self.word_vecs)
        # self.corr_loss = tf.sqrt(2 * tf.nn.l2_loss(self.emb_v - self.emb_w))
        # self.corr_loss = tf.reduce_mean(self.corr_loss)
        # dissimilar loss
        emb_v_ = tf.reduce_sum(self.emb_v, axis=1, keep_dims=True)
        emb_w_ = tf.reduce_sum(self.emb_w, axis=1, keep_dims=True)
        distance_map = tf.matmul(emb_v_, tf.ones([1, self.model_params.batch_size])) - tf.matmul(self.emb_v,
                                                                                                 tf.transpose(
                                                                                                     self.emb_w)) + \
                       tf.matmul(tf.ones([self.model_params.batch_size, 1]), tf.transpose(emb_w_))
        mask_initial = tf.to_float(
            tf.matmul(self.y_single, tf.ones([1, self.model_params.batch_size], dtype=tf.int32)) - \
            tf.matmul(tf.ones([self.model_params.batch_size, 1], dtype=tf.int32), tf.transpose(self.y_single)))
        mask = tf.to_float(tf.not_equal(mask_initial, tf.zeros_like(mask_initial)))
        masked_dissimilar_loss = tf.multiply(distance_map, mask)
        self.dissimilar_loss = tf.reduce_mean(tf.maximum(0., 0.1 * tf.ones_like(mask) - masked_dissimilar_loss))
        self.similar_loss = tf.sqrt(2 * tf.nn.l2_loss(self.emb_v - self.emb_w))
        self.similar_loss = tf.reduce_mean(self.similar_loss)
        logits_v = self.label_classifier(self.emb_v)
        logits_w = self.label_classifier(self.emb_w, reuse=True)
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_v) + \
                          tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_w)
        self.label_loss = tf.reduce_mean(self.label_loss)
        self.emb_loss = 50 * self.label_loss + self.similar_loss + 0.2 * self.dissimilar_loss
        self.emb_v_class = self.domain_classifier(self.emb_v, self.l)
        self.emb_w_class = self.domain_classifier(self.emb_w, self.l, reuse=True)

        all_emb_v = tf.concat([tf.ones([self.model_params.batch_size, 1]),
                               tf.zeros([self.model_params.batch_size, 1])], 1)
        all_emb_w = tf.concat([tf.zeros([self.model_params.batch_size, 1]),
                               tf.ones([self.model_params.batch_size, 1])], 1)
        self.domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_v_class, labels=all_emb_w) + \
                                 tf.nn.softmax_cross_entropy_with_logits(logits=self.emb_w_class, labels=all_emb_v)
        self.domain_class_loss = tf.reduce_mean(self.domain_class_loss)
        self.t_vars = tf.trainable_variables()
        self.vf_vars = [v for v in self.t_vars if 'vf_' in v.name]
        self.le_vars = [v for v in self.t_vars if 'le_' in v.name]
        self.dc_vars = [v for v in self.t_vars if 'dc_' in v.name]
        self.lc_vars = [v for v in self.t_vars if 'lc_' in v.name]

    def visual_feature_embed(self, X, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(X, 4096, scope='vf_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 1000, scope='vf_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='vf_fc_2'))
        return net

    def label_embed(self, L, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(L, 1000, scope='le_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 300, scope='le_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='le_fc_2'))
        return net

    def label_classifier(self, X, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = slim.fully_connected(X, 10, scope='lc_fc_0')
        return net

    def domain_classifier(self, E, l, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            E = flip_gradient(E, l)
            net = slim.fully_connected(E, self.model_params.semantic_emb_dim / 2, scope='dc_fc_0')
            net = slim.fully_connected(net, self.model_params.semantic_emb_dim / 4, scope='dc_fc_1')
            net = slim.fully_connected(net, 2, scope='dc_fc_2')
        return net

    def train(self, sess):
        emb_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_emb,
            beta1=0.5).minimize(self.emb_loss, var_list=self.le_vars + self.vf_vars)
        domain_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_domain,
            beta1=0.5).minimize(self.domain_class_loss, var_list=self.dc_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()
        self.model_params.epoch = 50
        start_time = time.time()
        for epoch in range(self.model_params.epoch):

            p = float(epoch) / self.model_params.epoch
            l = 2. / (1. + np.exp(-10. * p)) - 1
            for batch_feat, batch_vec, batch_labels, batch_labels_single, idx in self.data_iter.train_data():
                # sess.run([total_train_op], feed_dict={self.visual_feats: batch_feat, self.word_vecs: batch_vec, self.y: b,self.l: l})
                sess.run([emb_train_op, domain_train_op],
                         feed_dict={
                             self.visual_feats: batch_feat,
                             self.word_vecs: batch_vec,
                             self.y: batch_labels,
                             self.y_single: batch_labels_single[:, np.newaxis],
                             self.l: l})

                label_loss_val, similar_loss_val, emb_loss_val, domain_loss_val, dissimilar_loss_val = sess.run(
                    [self.label_loss, self.similar_loss, self.emb_loss, self.domain_class_loss, self.dissimilar_loss],
                    feed_dict={self.visual_feats: batch_feat,
                               self.word_vecs: batch_vec,
                               self.y: batch_labels,
                               self.y_single: batch_labels_single[:, np.newaxis],
                               self.l: l})
                print(
                    'Epoch: [%2d][%4d/%4d] time: %4.4f, emb_loss: %.8f, domain_loss: %.8f, label_loss: %.8f, similar_loss: %.8f, suaimilar_loss: %.8f' % (
                        epoch, idx, self.data_iter.num_train_batch, time.time() - start_time, emb_loss_val,
                        domain_loss_val, label_loss_val, similar_loss_val, dissimilar_loss_val
                    ))


# if __name__ == '__main__':
    # tf.app.run()


