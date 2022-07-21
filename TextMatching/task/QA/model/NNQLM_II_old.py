#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
from tqdm import tqdm
# point_wise obbject
rng = np.random.RandomState(23455)
class NNQLM_II(object):
    def __init__(self, opt):
        self.dropout_keep_prob = opt.dropout_keep_prob
        self.num_filters = opt.num_filters
        self.embeddings = opt.embeddings
        self.embedding_size = opt.embedding_size
        self.overlap_needed = opt.overlap_needed
        self.vocab_size = opt.vocab_size
        self.trainable = opt.trainable
        self.filter_sizes = opt.filter_sizes
        self.kernel_sizes = [3,4,5]
        self.pooling = opt.pooling
        self.position_needed = opt.position_needed
        if self.overlap_needed:
            self.total_embedding_dim = opt.embedding_size + opt.extend_feature_dim
        else:
            self.total_embedding_dim = opt.embedding_size
        if self.position_needed:
            self.total_embedding_dim = self.total_embedding_dim + opt.extend_feature_dim
        self.batch_size = opt.batch_size
        self.l2_reg_lambda = opt.l2_reg_lambda
        self.para = []
        self.max_input_left = opt.max_input_left
        self.max_input_right = opt.max_input_right
        self.hidden_num = opt.hidden_num
        self.extend_feature_dim = opt.extend_feature_dim
        self.is_Embedding_Needed = opt.is_Embedding_Needed
        self.rng = 23455
    def create_placeholder(self):
        self.question = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'input_answer')
        self.input_y = tf.placeholder(tf.float32, [None,2], name = "input_y")
        self.q_overlap = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_feature_embeding')
        self.a_overlap = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_feature_embeding')
        self.q_position = tf.placeholder(tf.int32,[None,self.max_input_left],name = 'q_position')
        self.a_position = tf.placeholder(tf.int32,[None,self.max_input_right],name = 'a_position')
    def density_weighted(self):
        self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left,1,1]), name = 'weighted_q')
        self.para.append(self.weighted_q)
        self.weighted_a = tf.Variable(tf.ones([1,self.max_input_right,1,1]) , name = 'weighted_a')
        self.para.append(self.weighted_a)
        self.word_weighted_q = tf.Variable(tf.ones([self.batch_size,self.embedding_size,1]), name = 'word_weighted_q')
        self.para.append(self.word_weighted_q)
        self.word_weighted_a = tf.Variable(tf.ones([self.batch_size,self.embedding_size,1]), name = 'word_weighted_a')
        self.para.append(self.word_weighted_a)
        # self.weighted_q_diff = tf.Variable(tf.ones([1,(self.max_input_left*self.max_input_left - self.max_input_left) / 2,1,1]), name = 'weighted_q_diff')
        # self.para.append(self.weighted_q_diff)
        # self.weighted_a_diff = tf.Variable(tf.ones([1,(self.max_input_right*self.max_input_right - self.max_input_right) / 2,1,1]) , name = 'weighted_a_diff')
        # self.para.append(self.weighted_a_diff)
        # self.weight_word = tf.Variable(tf.ones([self.batch_size,2,1,1]), name = 'weighted_word')
        # self.para.append(self.weight_word)
        # self.query_gate_weight = tf.Variable(tf.random_uniform([self.batch_size,self.max_input_left,1],minval=-0.5, maxval=0.5, dtype=tf.float32), trainable = True, name = "char_embedding")
        # self.para.append(self.query_gate_weight)

    def add_embeddings(self):

        # Embedding layer for both CNN
        with tf.name_scope("embedding"):
            if self.is_Embedding_Needed:
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = self.trainable )
            else:
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
            self.embedding_W = W
            self.overlap_W =  tf.get_variable('overlap_w', shape=[3, self.embedding_size],initializer = tf.random_normal_initializer())#tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = True)
            self.position_W = tf.Variable(tf.random_uniform([300,self.embedding_size], -1.0, 1.0),name = 'W',trainable = True)

            # self.weight_word = tf.get_variable("weighted_word", shape = [self.batch_size,2,1,1], initializer = tf.random_normal_initializer())
            # self.para.append(self.embedding_W) 
            # self.para.append(self.overlap_W)
        #get embedding from the word indices
        self.embedded_chars_q = self.concat_embedding(self.question,self.q_overlap,self.q_position)
        self.embedded_chars_a = self.concat_embedding(self.answer,self.a_overlap,self.a_position)

        # reduce_matrix
        self.embedded_reduce_q = tf.squeeze(self.embedded_chars_q, -1)
        self.embedded_reduce_a = tf.squeeze(self.embedded_chars_a, -1)

    def joint_representation(self):
        self.density_q = self.density_matrix(self.embedded_chars_q,self.weighted_q)
        # orginal
        self.density_a = self.density_matrix(self.embedded_chars_a,self.weighted_a)

        # self.q_all = tf.reshape(self.embedded_reduce_q, [self.batch_size,-1])

        # self.a_all = tf.reshape(self.embedded_reduce_a, [self.batch_size,-1]) 

        self.q_tmp = tf.squeeze(tf.matmul(self.embedded_reduce_q, self.word_weighted_q), -1)

        self.q_alp = tf.expand_dims(tf.nn.softmax(self.q_tmp, -1), 2)

        self.q_all = tf.reduce_sum(tf.multiply(self.embedded_reduce_q, self.q_alp), 1)

        self.a_tmp = tf.squeeze(tf.matmul(self.embedded_reduce_a, self.word_weighted_a), -1)

        self.a_alp = tf.expand_dims(tf.nn.softmax(self.a_tmp, -1), 2)

        self.a_all = tf.reduce_sum(tf.multiply(self.embedded_reduce_a, self.a_alp), 1)

        # self.M_qa = tf.matmul(self.density_q, self.temp_all_composite_matrix)
        self.M_qa = tf.matmul(self.density_q, self.density_a)
        # new 
        # self.density_a_tmp = self.composite_and_partialTrace()
        # self.density_a = self.conv2_out

        # self.density_a = self.conv2_out
        # self.M_qa = tf.matmul(self.density_q,self.reduced_matrix)
        # tmp = tf.matmul(self.density_q,self.density_a)
        # self.M_qa = tf.multiply(tmp,self.query_gate_weight)
        # self.reduced_matrix_information = self.get_information_from_reduced_matrix(self.reduced_matrix)

        # self.M_qa_2 = tf.matmul(self.embedded_reduce_q, self.conv2_out)
        # self.M_qa_2 = tf.multiply(self.M_qa_2, self.query_gate_weight)
        # self.M_qa_2 = tf.reshape(self.M_qa_2, [self.batch_size, -1])

    # def direct_representation(self):

    #     self.embedded_q = tf.reshape(self.embedded_chars_q,[-1,self.max_input_left,self.total_embedding_dim])
    #     self.embedded_a = tf.reshape(self.embedded_chars_a,[-1,self.max_input_right,self.total_embedding_dim])
    #     reverse_a = tf.transpose(self.embedded_a,[0,2,1])
    #     self.M_qa = tf.matmul(self.embedded_q,reverse_a)

    def trace_represent(self):
        # self.density_diag = tf.matrix_diag_part(self.M_qa)
        # self.density_trace = tf.expand_dims(tf.trace(self.M_qa),-1)
        # self.match_represent = tf.concat([self.density_diag,self.density_trace],1)
        self.density_diag = tf.matrix_diag_part(self.density_a)
        self.density_trace = tf.expand_dims(tf.trace(self.density_a),-1)
        self.match_represent = tf.concat([self.density_diag,self.density_trace],1)

    def composite_and_partialTrace(self):
        print ("creat composite and calculate partial trace!!")

        self.normal_embedded_q = tf.nn.l2_normalize(self.embedded_reduce_q,2) 
        self.normal_embedded_d = tf.nn.l2_normalize(self.embedded_reduce_a,2)

        embedded_chars_q1 = tf.expand_dims(self.normal_embedded_q,-1)
        embedded_chars_d1_T = tf.transpose(tf.expand_dims(self.normal_embedded_d,-1),perm = [0,1,3,2])

        #j-th word in query
        # for j in range(self.max_input_left):
            
        #     j_th_word = tf.tile(tf.expand_dims(tf.expand_dims(self.normal_embedded_q[:,j,:],-1), 1), [1,self.max_input_left,1,1])

        #     j_th_word_T = tf.transpose(j_th_word, perm = [0,1,3,2])

        #     if j == 0:
        #         self.all_query_dot = tf.matmul(j_th_word_T,embedded_chars_q1)
        #     else:
        #         self.all_query_dot = tf.concat([self.all_query_dot,tf.matmul(j_th_word_T,embedded_chars_q1)],1)  # [:,i:,:,:]
  
        # # self.all_query_dot = tf.multiply(self.all_query_dot, self.all_query_weight)

        # self.all_query_dot = tf.reduce_sum(self.all_query_dot, 1)

        self.temp_all_composite_matrix = tf.zeros([self.batch_size, self.embedding_size, self.embedding_size])

        self.temp_all_composite_matrix2 = tf.zeros([self.batch_size, self.embedding_size, self.embedding_size])

        #i-th word in document      
        for i in tqdm(range(self.max_input_right)):
            
            i_th_word = tf.tile(tf.expand_dims(tf.expand_dims(self.normal_embedded_d[:,i,:],-1), 1), [1,self.max_input_right,1,1])

            d_dT = tf.matmul(i_th_word,embedded_chars_d1_T)

            for k in range(self.max_input_right):

                # mul1 = tf.multiply(d_dT[:,k,:,:], self.all_query_dot)

                if k == 0:                  
                    self.temp_all_composite_matrix = d_dT[:,k,:,:]
                else:                   
                    self.temp_all_composite_matrix = tf.add(self.temp_all_composite_matrix, d_dT[:,k,:,:])

                # if k == i:

                #     self.temp_all_composite_matrix = tf.add(self.temp_all_composite_matrix, d_dT[:,k,:,:])

                # if k > i:

                #     self.temp_all_composite_matrix2 = tf.add(self.temp_all_composite_matrix2, d_dT[:,k,:,:])

            if i == 0:
                self.reduced_matrix1 =  self.temp_all_composite_matrix
            else:
                self.reduced_matrix1 = tf.add(self.reduced_matrix1,self.temp_all_composite_matrix)

            # self.reduced_matrix1 = tf.concat([tf.expand_dims(self.temp_all_composite_matrix, 1), tf.expand_dims(self.temp_all_composite_matrix2, 1)], 1)

            # self.reduced_matrix1 = tf.multiply(self.reduced_matrix1, self.weight_word)

            # self.reduced_matrix1 = tf.reduce_sum(self.reduced_matrix1, 1)

            # self.reduced_matrix1 = tf.multiply(self.reduced_matrix1, self.all_query_dot)

        self.reduced_matrix = self.reduced_matrix1

    def get_information_from_reduced_matrix(self, input_matrix):
        unit_tensor = tf.eye(input_matrix.get_shape()[1].value, batch_shape=[self.batch_size])
        diag_tensor = tf.matmul(input_matrix,unit_tensor)
        for i in range(self.batch_size):
            single_matrix = input_matrix[i]
            if i == 0:
                self.reduced_matrix_diag = tf.expand_dims(tf.diag_part(single_matrix),0)
                self.reduced_matrix_trace = tf.expand_dims(tf.trace(single_matrix),0)
            else:
                self.reduced_matrix_diag = tf.concat([self.reduced_matrix_diag,tf.expand_dims(tf.diag_part(single_matrix),0)],0)
                self.reduced_matrix_trace = tf.concat([self.reduced_matrix_trace,tf.expand_dims(tf.trace(single_matrix),0)],0)
        return tf.concat([self.reduced_matrix_diag, tf.expand_dims(self.reduced_matrix_trace,1)],1)

    #construct the density_matrix
    def density_matrix(self,sentence_matrix,sentence_weighted):
        # print sentence_matrix
        # print tf.nn.l2_normalize(sentence_matrix,2)

        self.norm = tf.nn.l2_normalize(sentence_matrix,2)
        # self.norm = tf.nn.softmax(sentence_matrix,2)
        # self.norm = sentence_matrix
        # print tf.reduce_sum(norm,2)
        reverse_matrix = tf.transpose(self.norm, perm = [0,1,3,2])
        q_a = tf.matmul(self.norm,reverse_matrix)
        # return tf.reduce_sum(tf.matmul(self.norm,reverse_matrix), 1)
        sam = tf.reduce_sum(tf.multiply(q_a,sentence_weighted),1)

        # for i in range(sentence_matrix.get_shape()[1]):
        # 	for j in range(sentence_matrix.get_shape()[1]):
        # 		if i == 0 and j == 1:
        # 			tmp = tf.expand_dims(tf.matmul(self.norm[:,i,:,:], reverse_matrix[:,j,:,:]), 1)
        # 		elif j > i:
        # 			tmp = tf.concat([tmp, tf.expand_dims(tf.matmul(self.norm[:,i,:,:], reverse_matrix[:,j,:,:]), 1)], 1)

        # diff = tf.reduce_sum(tf.multiply(tmp, comb_weighted), 1)

        return sam


    def feed_neural_work(self):     
        with tf.name_scope('regression'):
            #W = tf.Variable(tf.zeros(shape = [(self.total_embedding_dim - self.filter_sizes[0] + 1) * self.num_filters * 2,2]),name = 'W') 
            W = tf.Variable(tf.zeros(shape = [(self.embedding_size - self.filter_sizes[0] + 1) * self.num_filters * 2 + self.embedding_size * 2,2]),name = 'W') 
            # W = tf.Variable(tf.zeros(shape = [18900,2]),name = 'W')
            # W = tf.Variable(tf.zeros(shape = [self.max_input_left * self.embedding_size + (self.embedding_size - self.filter_sizes[0] + 1) * self.num_filters * 2, 2]),name = 'W')
        # with tf.name_scope('neural_network'):
        #     W = tf.get_variable(
        #         "W_hidden",
        #         shape=[(self.total_embedding_dim - self.filter_sizes[0] + 1) * self.num_filters * 2,self.hidden_num],
        #         # shape = [self.total_embedding_dim + 1,self.hidden_num],
        #         initializer = tf.contrib.layers.xavier_initializer())
        #     b = tf.get_variable('b_hidden', shape=[self.hidden_num],initializer = tf.random_normal_initializer())
        #     self.para.append(W)
        #     self.para.append(b)
        #     self.hidden_output = tf.nn.tanh(tf.nn.xw_plus_b(self.represent, W, b, name = "hidden_output"))
        # #add dropout
        # with tf.name_scope('dropout'):
        #     self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
        # with tf.name_scope("output"):
        #     W = tf.get_variable(
        #         "W_output",
        #         shape = [self.hidden_num, 2],
        #         initializer = tf.contrib.layers.xavier_initializer())
        #     b = tf.get_variable('b_output', shape=[2],initializer = tf.random_normal_initializer())
            b = tf.Variable(tf.zeros([2]), name = 'b')
            self.para.append(W)
            self.para.append(b)
            # self.conv2_out = tf.reshape(self.conv2_out, [self.batch_size, -1])
            # self.represent = tf.concat([self.represent, self.conv2_out], -1)
            # print(self.represent) # [batch, 7680]
            # print(self.match_represent) # [batch, 51]
            # self.represent = tf.concat([self.represent, self.M_qa_2], -1)
            # self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "scores")
            # self.represent = tf.nn.dropout(self.represent, self.dropout_keep_prob, name="hidden_output_drop")
            self.represent = tf.concat([self.represent, self.q_all, self.a_all], -1)
            self.represent = tf.nn.dropout(self.represent, self.dropout_keep_prob, name="hidden_output_drop")
            self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "scores")
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
            
    def create_loss(self):
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            #pi_regularization = tf.reduce_sum(self.weighted_q) - 1 + tf.reduce_sum(self.weighted_a) - 1
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    # def concat_embedding(self,words_indice,overlap_indice,position_indice):
    #     embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
    #     position_embedding = tf.nn.embedding_lookup(self.position_W,position_indice)
    #     overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
    #     if not self.overlap_needed :
    #         if not self.position_needed:
    #             return tf.expand_dims(embedded_chars_q,-1)
    #         else:
    #             return tf.expand_dims(tf.reduce_sum([embedded_chars_q,position_embedding],0),-1)
    #     else:
    #         if not self.position_needed:
    #             return  tf.expand_dims(tf.reduce_sum([embedded_chars_q,overlap_embedding_q],0),-1)
    #         else:
    #             return tf.expand_dims(tf.reduce_sum([embedded_chars_q,overlap_embedding_q,position_embedding],0),-1)


    def concat_embedding(self,words_indice,overlap_indice,position_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        position_embedding = tf.nn.embedding_lookup(self.position_W,position_indice)
        # overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        return tf.expand_dims(embedded_chars_q,-1)
        # else:
        #     return tf.expand_dims(tf.reduce_sum([embedded_chars_q,position_embedding],0),-1)
        
    def convolution(self):
        #initialize my conv kernel
        self.kernels = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size,filter_size,1,self.num_filters]
                fan_in = np.prod(filter_shape[:-1])
                fan_out = (filter_shape[-1] * np.prod(filter_shape[:2]))
                W_bound = np.sqrt(6. / (fan_in + fan_out))

                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "W")
                # W = tf.Variable(tf.random_uniform(filter_shape,minval = -W_bound,maxval = W_bound,seed = self.rng))
                W = tf.Variable(np.asarray(rng.uniform(low = -W_bound,high = W_bound,size = filter_shape),dtype = 'float32'))
                b = tf.Variable(tf.constant(0.0, shape = [self.num_filters]), name = "b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        # self.qa_0, self.qa_1, self.qa_2 = self.narrow_convolution(tf.expand_dims(self.M_qa,-1))
       	self.qa = self.narrow_convolution(tf.expand_dims(self.M_qa,-1))

    def ngram_cnn_network(self):
        self.CNN_input = tf.expand_dims(self.reduced_matrix,-1)
        with tf.variable_scope("ngram_cnn",reuse=tf.AUTO_REUSE):
            # cnn layer1
            conv_outs = []
            for size in self.kernel_sizes:

                conv_matrix = tf.layers.conv2d(self.CNN_input, 16, size, strides = 1, padding='SAME')               

                max_pool = tf.layers.max_pooling2d(conv_matrix, pool_size=[2,2],strides=[1,1],padding='SAME')

                conv_outs.append(max_pool)
            
            for i in range(len(self.kernel_sizes)):
                if i == 0:
                    self.conv_pooling_outs = conv_outs[i]
                else:
                    self.conv_pooling_outs = tf.concat([self.conv_pooling_outs, conv_outs[i]],-1)
            
            # cnn layer2
            conv_layer2 = tf.layers.conv2d(self.conv_pooling_outs, 1, 5, strides = 1, padding='SAME')
            self.conv2_max_pool = tf.layers.max_pooling2d(conv_layer2, pool_size=[2,2],strides=[1,1],padding='SAME')
            self.conv2_out = tf.reshape(self.conv2_max_pool,[self.batch_size,self.conv2_max_pool.get_shape()[1].value,-1])

        # return self.conv2_out

    def max_pooling(self,conv):
        pooled = tf.nn.max_pool(
                    conv,
                    # ksize = [1, self.total_embedding_dim, self.total_embedding_dim, 1],
                    ksize = [1, self.max_input_left, self.max_input_right, 1],
                    # ksize = [1, 5, 5, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled
    def pooling_graph(self):
        with tf.name_scope('pooling'):
      
            # pooling_0 = self.max_pooling(self.qa_0)
            # pooling_1 = self.max_pooling(self.qa_1)
            # pooling_2 = self.max_pooling(self.qa_2)

            # pooling_0 = tf.reshape(pooling_0, [self.batch_size, -1])
            # pooling_1 = tf.reshape(pooling_1, [self.batch_size, -1])
            # pooling_2 = tf.reshape(pooling_2, [self.batch_size, -1])

            # self.represent = tf.concat([pooling_0,pooling_1,pooling_2],1)
            # print self.pooling
            # self.represent = tf.reshape(pooling,[-1,self.num_filters * len(self.filter_sizes)])

            for i in range(len(self.qa)):

                raw_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qa[i],1))

                col_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qa[i],2))

                if i == 0:

                    tmp = tf.concat([raw_pooling,col_pooling],1)

                else:

                    tmp = tf.concat([tmp, tf.concat([raw_pooling,col_pooling],1)], 1)

            self.represent = tmp
            

    def wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name="conv-1"
            )

            h = tf.nn.tanh(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(cnn_outputs,3)
        return cnn_reshaped
    def narrow_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )
            self.see = conv
            h = tf.nn.tanh(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        # cnn_reshaped = tf.concat(cnn_outputs,3)
        # return cnn_outputs[0], cnn_outputs[1], cnn_outputs[2]
        # return cnn_reshaped
        return cnn_outputs

    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.density_weighted()
        # self.composite_and_partialTrace()
        # self.ngram_cnn_network()
        self.joint_representation()
        # self.ngram_cnn_network()
        # self.direct_representation()
        # self.trace_represent()
        self.convolution()
        self.pooling_graph()
        # self.interact()
        self.feed_neural_work()
        self.create_loss()

# if __name__ == '__main__':
#     cnn = QA_quantum(max_input_left = 33,
#                 max_input_right = 40,
#                 vocab_size = 5000,
#                 embedding_size = 50,
#                 batch_size = 3,
#                 embeddings = None,
#                 dropout_keep_prob = 1,
#                 filter_sizes = [40],
#                 num_filters = 65,
#                 l2_reg_lambda = 0.0,
#                 is_Embedding_Needed = False,
#                 trainable = True,
#                 overlap_needed = False,
#                 pooling = 'max',
#                 position_needed = False)
#     cnn.build_graph()
#     input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
#     input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
#     input_y = np.ones((3,2))

#     input_overlap_q = np.ones((3,33))
#     input_overlap_a = np.ones((3,40))
#     q_posi = np.ones((3,33))
#     a_posi = np.ones((3,40))
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         feed_dict = {
#             cnn.question:input_x_1,
#             cnn.answer:input_x_2,
#             cnn.input_y:input_y,
#             cnn.q_overlap:input_overlap_q,
#             cnn.a_overlap:input_overlap_a,
#             cnn.q_position:q_posi,
#             cnn.a_position:a_posi
#         }
       
#         see,question,answer,scores = sess.run([cnn.embedded_chars_q,cnn.question,cnn.answer,cnn.scores],feed_dict)
#         # print see

       
