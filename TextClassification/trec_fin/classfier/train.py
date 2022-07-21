# coding=utf-8
#! /usr/bin/env python3.4
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from helper_my import load_wiki,get_overlap_dict, batch_gen_with_point_wise, load, prepare, batch_gen_with_single,test_my,load_ag_news,load_trec_sst2
import operator
# from model.QA_CNN_point import QA_quantum as model
# from model.CNNQLM_II_flat import CNN as model
# from model.vocab import CNN as model
# from model.dim   import CNN as model
# from model.CNNQLM_I   import CNN as model
import random
import evaluation
from sklearn.metrics import accuracy_score
import pickle
import config
from functools import wraps
import opts
import model 
from model.__init__ import setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
now = int(time.time())
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

from functools import wraps

FLAGS = config.flags.FLAGS
# FLAGS._parse_flags()
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
log_dir = 'qa_log/' + timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
para_file = log_dir + '/test_' + FLAGS.data + timeStamp + '_para'
precision = data_file + 'precise'


if FLAGS.data=='TREC' or FLAGS.data=='sst2':
    train,dev,test=load_trec_sst2(FLAGS.data)
elif FLAGS.data=='ag_news':
    train, dev = load_ag_news(FLAGS.data)
else:
    train, dev = load(FLAGS.data)
# q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
# a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
q_max_sent_length = 22
a_max_sent_length = 30
alphabet, embeddings,embeddings_complex = prepare(
        [train,  dev], max_sent_length=a_max_sent_length,dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
opt = opts.parse_opt(q_max_sent_length,a_max_sent_length,alphabet,embeddings,embeddings_complex)

def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco
#@log_time_delta
def predict(sess, cnn, dev, alphabet, batch_size, q_len):
    scores = []
    for data in batch_gen_with_single(dev, alphabet, batch_size, q_len):
        feed_dict = {
            cnn.question: data[0],
            cnn.q_position: data[1],
            cnn.dropout_keep_prob: 1.0
        }
        score = sess.run(cnn.scores, feed_dict)
        scores.extend(score)
    return np.array(scores[:len(dev)])

@log_time_delta
def test_point_wise():
    # train, dev, test = load_wiki(FLAGS.data, filter=FLAGS.clean)  #wiki
    # # train, test, dev = load(FLAGS.data, filter=FLAGS.clean) #trec
    # q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    # a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print(q_max_sent_length)
    print(a_max_sent_length)
    print(len(train))
    print ('train question unique:{}'.format(len(train['question'].unique())))
    print ('train length', len(train))
    # print ('test length', len(test))
    print ('dev length', len(dev))

    # alphabet, embeddings,embeddings_complex = prepare(
        # [train, test, dev], max_sent_length=a_max_sent_length,dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
    print(embeddings_complex)
    print ('alphabet:', len(alphabet))
    # opt = opts.parse_opt(q_max_sent_length,a_max_sent_length,alphabet,embeddings,embeddings_complex)
    with tf.Graph().as_default():
        with tf.device("/gpu:2"):
            session_conf = tf.ConfigProto()
            session_conf.allow_soft_placement = FLAGS.allow_soft_placement
            session_conf.log_device_placement = FLAGS.log_device_placement
            session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default(), open(precision, "w") as log:
            s='embedding_dim:  '+str(FLAGS.embedding_dim)+'\n'+'dropout_keep_prob:  '+str(FLAGS.dropout_keep_prob)+'\n'+'l2_reg_lambda:  '+str(FLAGS.l2_reg_lambda)+'\n'+'learning_rate:  '+str(FLAGS.learning_rate)+'\n'+'batch_size:  '+str(FLAGS.batch_size)+'\n''trainable:  '+str(FLAGS.trainable)+'\n'+'num_epochs:  '+str(FLAGS.num_epochs)+'\n''data:  '+str(FLAGS.data)+'\n'
            log.write(str(s) + '\n')
            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            # cnn = model(opt
            #     max_input_left=q_max_sent_length,
            #     max_input_right=a_max_sent_length,
            #     vocab_size=len(alphabet),
            #     embedding_size=FLAGS.embedding_dim,
            #     batch_size=FLAGS.batch_size,
            #     embeddings=embeddings,
            #     embeddings_complex=embeddings_complex,
            #     dropout_keep_prob=FLAGS.dropout_keep_prob,
            #     filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            #     num_filters=FLAGS.num_filters,
            #     l2_reg_lambda=FLAGS.l2_reg_lambda,
            #     is_Embedding_Needed=True,
            #     trainable=FLAGS.trainable,
            #     overlap_needed=FLAGS.overlap_needed,
            #     position_needed=FLAGS.position_needed,
            #     pooling=FLAGS.pooling,
            #     hidden_num=FLAGS.hidden_num,
            #     extend_feature_dim=FLAGS.extend_feature_dim)
            cnn = setup(opt)
            cnn.build_graph()
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            starter_learning_rate = FLAGS.learning_rate
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step, 100, 0.96)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            # optimizer =  tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            sess.run(tf.global_variables_initializer())
            acc_max = 0.0000
            now = int(time.time())
            timeArray = time.localtime(now)
            timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
            timeDay = time.strftime("%Y%m%d", timeArray)
            print (timeStamp)
            for i in range(FLAGS.num_epochs):
                datas = batch_gen_with_point_wise(
                    train, alphabet, FLAGS.batch_size, q_len=q_max_sent_length)
                for data in datas:
                    feed_dict = {
                        cnn.question: data[0],
                        cnn.input_y: data[1],
                        cnn.q_position: data[2],
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    start = time.time()
                    _, step, loss, accuracy = sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    # print("======================batch_over======================")
                    print("{}: step {}, loss {:g}, acc {:g}  in {:.4f} seconds ".format(time_str, step, loss, accuracy,time.time()-start))
                predicted = predict(sess, cnn, train, alphabet, FLAGS.batch_size, q_max_sent_length)
                predicted_label = np.argmax(predicted, 1)
               # print("-"*10)
               # print(predicted_label)
                acc_train= accuracy_score(predicted_label,train['flag'])
                predicted_dev = predict(sess, cnn, dev, alphabet, FLAGS.batch_size, q_max_sent_length)
                predicted_label = np.argmax(predicted_dev, 1)
                acc_dev= accuracy_score(predicted_label,dev['flag'])
                if acc_dev> acc_max:
                    tf.train.Saver().save(sess, "model_save/model",write_meta_graph=True)
                    acc_max = acc_dev
                print ("{}:train epoch:acc {}".format(i, acc_train))
                print ("{}:dev epoch:acc {}".format(i, acc_dev))
                line2 = " {} epoch:train_acc:{}".format(i, acc_train)
                line3 = " {} epoch:dev_acc:{}".format(i, acc_dev)
                log.write(line2 + '\n' + line3 + '\n')
                log.flush()
            # acc_flod.append(acc_max)
            log.close()


if __name__ == '__main__':
    # test_quora()
    if FLAGS.loss == 'point_wise':
        test_point_wise()
    # test_pair_wise()
    # test_point_wise()
