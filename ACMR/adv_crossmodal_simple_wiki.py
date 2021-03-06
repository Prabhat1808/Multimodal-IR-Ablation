from __future__ import print_function

import cPickle
import os
import time

import numpy as np
import sklearn.preprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim

from base_model import BaseModel, BaseModelParams, BaseDataIter
# import utils
from flip_gradient import flip_gradient


class DataIter(BaseDataIter):
    def __init__(self, batch_size, data):
        BaseDataIter.__init__(self, batch_size)
        self.num_train_batch = 0
        self.num_test_batch = 0

        self.train_img_feats = data.x_train['img_train']
        self.train_txt_vecs = data.x_train['txt_train']
        self.train_labels = data.y_train['train_labels']
        self.test_img_feats = data.x_test['img_test']
        self.test_txt_vecs = data.x_test['txt_test']
        self.test_labels = data.y_test['test_labels']
        self.num_train_batch = len(self.train_img_feats) / self.batch_size
        self.num_test_batch = len(self.test_img_feats) / self.batch_size

    def train_data(self):
        for i in range(self.num_train_batch):
            batch_img_feats = self.train_img_feats[i * self.batch_size: (i + 1) * self.batch_size]
            batch_txt_vecs = self.train_txt_vecs[i * self.batch_size: (i + 1) * self.batch_size]
            batch_labels = self.train_labels[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_img_feats, batch_txt_vecs, batch_labels, i

    def test_data(self):
        for i in range(self.num_test_batch):
            batch_img_feats = self.test_img_feats[i * self.batch_size: (i + 1) * self.batch_size]
            batch_txt_vecs = self.test_txt_vecs[i * self.batch_size: (i + 1) * self.batch_size]
            batch_labels = self.test_labels[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch_img_feats, batch_txt_vecs, batch_labels, i


class ModelParams(BaseModelParams):
    def __init__(self):
        BaseModelParams.__init__(self)

        self.epoch = 200
        self.batch_size = 64
        self.visual_feat_dim = 4096
        # self.word_vec_dim = 300
        self.word_vec_dim = 5000
        self.lr_total = 0.0001
        self.lr_emb = 0.0001
        self.lr_domain = 0.0001
        self.top_k = 50
        self.semantic_emb_dim = 40
        self.dataset_name = 'wikipedia_dataset'
        self.model_name = 'adv_semantic_zsl'
        self.model_dir = 'adv_semantic_zsl_%d_%d_%d' % (self.visual_feat_dim, self.word_vec_dim, self.semantic_emb_dim)

        self.checkpoint_dir = 'checkpoint'
        self.sample_dir = 'samples'
        self.dataset_dir = './data'
        self.log_dir = 'logs'

    def update(self):
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)


class AdvCrossModalSimple(BaseModel):
    def __init__(self, model_params, dataIter):
        BaseModel.__init__(self, model_params)
        # self.data_iter = DataIter(self.model_params.batch_size)
        self.data_iter = dataIter
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
        # self.similar_loss = tf.reduce_mean(tf.abs(distance_map-masked_dissimilar_loss))
        self.similar_loss = tf.sqrt(2 * tf.nn.l2_loss(self.emb_v - self.emb_w))
        self.similar_loss = tf.reduce_mean(self.similar_loss)
        logits_v = self.label_classifier(self.emb_v)
        logits_w = self.label_classifier(self.emb_w, reuse=True)
        self.label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_v) + \
                          tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits_w)
        self.label_loss = tf.reduce_mean(self.label_loss)
        self.emb_loss = 100 * self.label_loss + self.similar_loss + 0.02 * self.dissimilar_loss
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
            net = tf.nn.tanh(slim.fully_connected(X, 512, scope='vf_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='vf_fc_1'))
            net = tf.nn.tanh(slim.fully_connected(net, self.model_params.semantic_emb_dim, scope='vf_fc_2'))
        return net

    def label_embed(self, L, is_training=True, reuse=False):
        with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
            net = tf.nn.tanh(slim.fully_connected(L, self.model_params.semantic_emb_dim, scope='le_fc_0'))
            net = tf.nn.tanh(slim.fully_connected(net, 100, scope='le_fc_1'))
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
        # self.check_dirs()

        total_loss = self.emb_loss + self.domain_class_loss
        total_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_total,
            beta1=0.5).minimize(total_loss)
        emb_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_emb,
            beta1=0.5).minimize(self.emb_loss, var_list=self.le_vars + self.vf_vars)
        domain_train_op = tf.train.AdamOptimizer(
            learning_rate=self.model_params.lr_domain,
            beta1=0.5).minimize(self.domain_class_loss, var_list=self.dc_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

        start_time = time.time()
        map_avg_ti = []
        map_avg_it = []
        adv_loss = []
        emb_loss = []
        losses = []
        for epoch in range(self.model_params.epoch):
            if epoch % 1 == 0:
                adv = 0
                emb = 0
                test_img_feats_trans = []
                test_txt_vecs_trans = []
                test_labels = []
                p = float(epoch) / self.model_params.epoch
                l = 2. / (1. + np.exp(-10. * p)) - 1
                for feats, vecs, labels, i in self.data_iter.train_data():
                    labels_ = labels - np.ones_like(labels)
                    label_binarizer = sklearn.preprocessing.LabelBinarizer()
                    label_binarizer.fit(range(max(labels_) + 1))
                    b = label_binarizer.transform(labels_)
                    emb_loss_val, domain_loss_val = sess.run([self.emb_loss, self.domain_class_loss],
                                                             feed_dict={self.visual_feats: feats,
                                                                        self.word_vecs: vecs,
                                                                        self.y: b,
                                                                        self.y_single: np.transpose([labels]),
                                                                        self.l: l})
                    emb += emb_loss_val
                    adv += domain_loss_val
                emb = float(emb) / float(25)
                adv = float(adv) / float(25)
                adv_loss.append(adv)
                emb_loss.append(emb)

                for feats, vecs, labels, i in self.data_iter.test_data():
                    feats_trans = sess.run(self.emb_v, feed_dict={self.visual_feats: feats})
                    vecs_trans = sess.run(self.emb_w, feed_dict={self.word_vecs: vecs})
                    test_labels += labels
                    for ii in range(len(feats)):
                        test_img_feats_trans.append(feats_trans[ii])
                        test_txt_vecs_trans.append(vecs_trans[ii])
                test_img_feats_trans = np.asarray(test_img_feats_trans)
                test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)

                k = self.model_params.top_k
                avg_precs = []
                for i in range(len(test_txt_vecs_trans)):
                    query_label = test_labels[i]
                    # distances and sort by distances
                    wv = test_txt_vecs_trans[i]
                    diffs = test_img_feats_trans - wv
                    dists = np.linalg.norm(diffs, axis=1)
                    sorted_idx = np.argsort(dists)
                    # for each k do top-k
                    precs = []
                    for topk in range(1, k + 1):
                        hits = 0
                        top_k = sorted_idx[0: topk]
                        if query_label != test_labels[top_k[-1]]:
                            continue
                        for ii in top_k:
                            retrieved_label = test_labels[ii]
                            if retrieved_label == query_label:
                                hits += 1
                        precs.append(float(hits) / float(topk))
                    if len(precs) == 0:
                        precs.append(0)
                    avg_precs.append(np.average(precs))
                map_avg_ti.append(np.mean(avg_precs))

                avg_precs_ = []
                for i in range(len(test_img_feats_trans)):
                    query_img_feat = test_img_feats_trans[i]
                    ground_truth_label = test_labels[i]
                    # calculate distance and sort
                    diffs = test_txt_vecs_trans - query_img_feat
                    dists = np.linalg.norm(diffs, axis=1)
                    sorted_idx = np.argsort(dists)
                    # for each k in top-k
                    precs = []
                    for topk in range(1, k + 1):
                        hits = 0
                        top_k = sorted_idx[0: topk]
                        if ground_truth_label != test_labels[top_k[-1]]:
                            continue
                        for ii in top_k:
                            retrieved_label = test_labels[ii]
                            if ground_truth_label == retrieved_label:
                                hits += 1
                        precs.append(float(hits) / float(topk))
                    if len(precs) == 0:
                        precs.append(0)
                    avg_precs_.append(np.average(precs))
                map_avg_it.append(np.mean(avg_precs_))
            if epoch == (self.model_params.epoch - 1):
                with open('./data/wikipedia_dataset/map_avg_ti.pkl', 'wb') as f:
                    cPickle.dump(map_avg_ti, f, cPickle.HIGHEST_PROTOCOL)
                with open('./data/wikipedia_dataset/map_avg_it.pkl', 'wb') as f:
                    cPickle.dump(map_avg_it, f, cPickle.HIGHEST_PROTOCOL)
                with open('./data/wikipedia_dataset/adv_loss.pkl', 'wb') as f:
                    cPickle.dump(adv_loss, f, cPickle.HIGHEST_PROTOCOL)
                with open('./data/wikipedia_dataset/emb_loss.pkl', 'wb') as f:
                    cPickle.dump(emb_loss, f, cPickle.HIGHEST_PROTOCOL)
                print('dump map_avg')

            if epoch == 0:
                test_img_feats_trans = []
                test_txt_vecs_trans = []
                for feats, vecs, labels, i in self.data_iter.train_data():
                    feats_trans = sess.run(self.emb_v, feed_dict={self.visual_feats: feats})
                    vecs_trans = sess.run(self.emb_w, feed_dict={self.word_vecs: vecs})
                    for ii in range(len(feats)):
                        test_img_feats_trans.append(feats_trans[ii])
                        test_txt_vecs_trans.append(vecs_trans[ii])
                test_img_feats_trans = np.asarray(test_img_feats_trans)
                test_txt_vecs_trans = np.asarray(test_txt_vecs_trans)
                test_feats_trans = np.concatenate((test_img_feats_trans[0:1000], test_txt_vecs_trans[-1000:]))
                with open('./data/wikipedia_dataset/before_feats_transformed.pkl', 'wb') as f:
                    cPickle.dump(test_feats_trans, f, cPickle.HIGHEST_PROTOCOL)
                print('dump before')

            p = float(epoch) / self.model_params.epoch
            l = 2. / (1. + np.exp(-10. * p)) - 1
            for batch_feat, batch_vec, batch_labels, idx in self.data_iter.train_data():
                # create one-hot labels
                batch_labels_ = batch_labels - np.ones_like(batch_labels)
                label_binarizer = sklearn.preprocessing.LabelBinarizer()
                label_binarizer.fit(range(max(batch_labels_) + 1))
                b = label_binarizer.transform(batch_labels_)
                # _, label_loss_val, dissimilar_loss_val, similar_loss_val = sess.run([total_train_op, self.label_loss, self.dissimilar_loss, self.similar_loss], feed_dict={self.visual_feats: batch_feat, self.word_vecs: batch_vec, self.y: b, self.y_single: np.transpose([batch_labels]),self.l: l})
                sess.run([emb_train_op, domain_train_op],
                         feed_dict={
                             self.visual_feats: batch_feat,
                             self.word_vecs: batch_vec,
                             self.y: b,
                             self.y_single: np.transpose([batch_labels_]),
                             self.l: l})
                label_loss_val, similar_loss_val, emb_loss_val, domain_loss_val, dissimilar_loss_val = sess.run(
                    [self.label_loss, self.similar_loss, self.emb_loss, self.domain_class_loss, self.dissimilar_loss],
                    feed_dict={self.visual_feats: batch_feat,
                               self.word_vecs: batch_vec,
                               self.y: b,
                               self.y_single: np.transpose([batch_labels]),
                               self.l: l})
                print(
                    'Epoch: [%2d][%4d/%4d] time: %4.4f, emb_loss: %.8f, domain_loss: %.8f, label_loss: %.8f, similar_loss: %.8f, dissimilar_loss: %.8f' % (
                        epoch, idx, self.data_iter.num_train_batch, time.time() - start_time, emb_loss_val,
                        domain_loss_val, label_loss_val, similar_loss_val, dissimilar_loss_val
                    ))
                sum_loss = emb_loss_val + domain_loss_val + label_loss_val + similar_loss_val + dissimilar_loss_val
                losses.append(sum_loss)
        params = {'emb_v': self.emb_v, 'dataIter': self.data_iter, 'emb_w': self.emb_w,
                  'visual_feats': self.visual_feats, 'word_vecs': self.word_vecs}
        logs = None
        losses = losses
        return params, losses, logs
