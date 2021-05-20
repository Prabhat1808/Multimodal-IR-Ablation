import tensorflow as tf
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
from models.adv_crossmodal_simple_nuswide import AdvCrossModalSimple, ModelParams, DataIter
from Framework.dataset import Dataset
from Framework.model import Parameters, Model
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
    inverted = argmax(encoded[0])
    model.evaluate(encoded, encoded)
    model.save_stats("acmr_wikipedia_stats.npy")


if __name__ == '__main__':
    tf.app.run()