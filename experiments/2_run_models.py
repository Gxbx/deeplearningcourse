import sys, os
import argparse
import os
import numpy as np
import argparse

sys.path.append(os.getcwd())
import tensorflow as tf

# Importar modelos de redes neuronales
from models import LogisticClassifier
from models import MLP
from models import LSTM

# Datos
from data.EmojiDatasetWords import EmojiDatasetWords

#evaluation
from experiments.evaluator import Evaluator

# Program's arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_model_path', type=str, default='./saved/models')
parser.add_argument('--save_result_path', type=str, default='./saved/results')
args = parser.parse_args()

# Parametros de la red
max_seq_len = 10
state_size = 280
learning_rate = 0.1
batch_size = 140
display_step = 10
validation_step = 50
model_path = None
restore_model = False

batch_x = None
batch_y = None

file_name = '../data/spanish_emojis.csv'
dataset = EmojiDatasetWords()
#dataset.load_datasets()


def get_model(model_name):
    if model_name.lower() == 'logistic':
        return LogisticClassifier.LogisticClassifier
    if model_name.lower() == 'mlp':
        return MLP.MLP
    if model_name.lower() == 'lstm':
        return LSTM.LSTM
    return LSTM.LSTM


def run(args, model_name='logistic', report_result=True, epoc=2000):
    dataset_info = dataset.get_train_test_val_data()

    train_data = dataset_info[0]
    train_label = dataset_info[1]
    train_seqlen = dataset_info[2]
    test_data = dataset_info[3]
    test_label = dataset_info[4]
    test_seqlen = dataset_info[5]
    val_data = dataset_info[6]
    val_label = dataset_info[7]
    val_seqlen = dataset_info[8]

    vocab_size = dataset_info[9]

    num_classes = dataset.get_number_classes()

    model_class = get_model(model_name)
    model = model_class(max_seq_len, state_size, vocab_size, num_classes)

    tf.reset_default_graph()
    model.build_model()
    loss, optimizer = model.step_training(learning_rate =  learning_rate)
    val_precission = 0.0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)

        ckpt = None
        if args.save_model_path:
            ckpt = tf.train.get_checkpoint_state(args.save_model_path)
        if ckpt and ckpt.model_checkpoint_path and restore_model:
            saver.restore(sess, ckpt.model_checkpoint_path)

        training_precision, validation_precision, train_loss = [], [], []
        for step in range(epoc):
            batch_x, batch_y, batch_seqlen = dataset.next_train_balanced(batch_size=batch_size)
            sess.run(optimizer, feed_dict={model.x: batch_x,
                                           model.y: batch_y,
                                           model.batch_size: len(batch_seqlen)})

            if step % display_step == 0 and step > 0:
                ops = [model.precision, loss]
                precision, loss_ = sess.run(ops, feed_dict={model.x: batch_x,
                                                            model.y: batch_y,
                                                            model.batch_size: len(batch_seqlen)})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss_) + ", Training Precision= " + \
                      "{:.5f}".format(precision))
                training_precision.append((step, precision))
                train_loss.append((step, loss_))

            if step % validation_step == 0 and step > 0:
                val_precision, y_probs = sess.run([model.precision, model.probs], feed_dict={model.x: val_data,
                                                                                         model.y: val_label,
                                                                                         model.batch_size: len(batch_seqlen)})
                print("Step " + str(step) + ", Validation Precision= {:.5f}".format(val_precision))
                validation_precision.append((step, val_precision))

                if saver and args.save_model_path:
                    saver.save(sess, os.path.join(args.save_model_path, 'trained_model.ckpt'),
                               global_step = step)

                # report results during training:
                if report_results:
                    evaluator = Evaluator()
                    class_names = dataset.get_class_names()
                    y_actual = val_label
                    y_preds = np.argmax(y_probs, axis=1)
                    evaluator.save_confussion_matrix(y_actual, y_preds, class_names, normalize=True,
                                                     file_path='./saved/confussion_matrix/confussion_matrix_%s.png' %(model_name),
                                             title='Confussion matrix for %s at time %d' %(model_name.upper(), step))
                    evaluator.save_precision(training_precision, validation_precision,
                                             file_path='./saved/precision/precision_%s.png' %(model_name),
                                             title='%s model at step %d' %(model_name.upper(), step))
                    evaluator.save_loss(train_loss, file_path='./saved/loss/loss_training_%s.png' % (model_name),
                                        title='%s model at step %d' % (model_name.upper(), step))

    return val_precision

def run_best_precission():
    models = ['lstm','mlp', 'logistic']
    num_experiments = 5 
    results = {}
    #Evaluator


