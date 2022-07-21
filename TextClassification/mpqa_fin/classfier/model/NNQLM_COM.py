#!/usr/bin/env python
#encoding=utf-8
import config
import tensorflow as tf
import numpy as np
from tqdm import tqdm
# point_wise obbject
rng = np.random.RandomState(2021)
FLAGS = config.flags.FLAGS
FLAGS.flag_values_dict()
class NNQLM_COM(object):
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
        self.rng = 2021
        self.data = opt.data
    def create_placeholder(self):
        self.question = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'input_question')
        if self.data=='TREC':
            self.input_y = tf.placeholder(tf.float32, [self.batch_size,6], name = "input_y")
        elif self.data=='ag_news':
            self.input_y = tf.placeholder(tf.float32, [self.batch_size,4], name = "input_y")
        else:
            self.input_y = tf.placeholder(tf.float32, [self.batch_size,2], name = "input_y")
        self.q_lens = tf.placeholder(tf.int32,[self.batch_size],name = 'question_length')
        self.q_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'q_position')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    def density_weighted(self):
        #self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left*2-1,1,1]), name = 'weighted_q')
        self.weighted_q = tf.Variable(tf.ones([1,self.max_input_left,1,1]), name = 'weighted_q')
        self.para.append(self.weighted_q)

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
            

        #get embedding from the word indices
        self.embedded_chars_q = self.concat_embedding(self.question,self.q_lens)
        # reduce_matrix
        # self.embedded_reduce_q = tf.squeeze(self.embedded_chars_q, -1)
        self.embedded_chars_q = self.density_matrix(self.embedded_chars_q,self.weighted_q)
        # self.embedded_chars_q = self.composite_and_partialTrace(self.embedded_chars_q)
        self.q_trace = self.trace_represent(self.embedded_chars_q)
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
        return tf.reduce_sum(tf.multiply(q_a,sentence_weighted),1)

    def composite_and_partialTrace(self, sentence_matrix):
        print ("creat composite and calculate partial trace!!")

        sentence_matrix = tf.squeeze(sentence_matrix, -1)

        normal_embedded_q = tf.nn.l2_normalize(sentence_matrix,2) 

        max_len = sentence_matrix.get_shape()[1]

        embedded_chars_q1_T = tf.transpose(tf.expand_dims(sentence_matrix,-1),perm = [0,1,3,2])

        #i-th word in document      
        for i in tqdm(range(max_len)):
            
            i_th_word = tf.tile(tf.expand_dims(tf.expand_dims(normal_embedded_q[:,i,:],-1), 1), [1,max_len,1,1])

            d_dT = tf.matmul(i_th_word,embedded_chars_q1_T)

            for k in range(max_len):

                if k == 0:                  
                    temp_all_composite_matrix = d_dT[:,k,:,:]
                else:                   
                    temp_all_composite_matrix = tf.add(temp_all_composite_matrix, d_dT[:,k,:,:])

                # if k == i:

                #     self.temp_all_composite_matrix = d_dT[:,k,:,:]

                # if k > i:

                #     self.temp_all_composite_matrix = tf.add(self.temp_all_composite_matrix, d_dT[:,k,:,:])

            if i == 0:
                reduced_matrix1 =  temp_all_composite_matrix
            else:
                reduced_matrix1 = tf.add(reduced_matrix1, temp_all_composite_matrix)

        reduced_matrix = reduced_matrix1

        return reduced_matrix


    def feed_neural_work(self):     
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        with tf.name_scope('regression'):
            self.represent = tf.concat([self.represent,self.q_trace], -1)
            # W = tf.Variable(tf.zeros(shape = [(self.total_embedding_dim - self.filter_sizes[0] + 1) * self.num_filters * 2,2]),name = 'W') 
            # W = tf.Variable(tf.zeros(shape = [self.max_input_left*self.embedding_size + 51 ,2]),name = 'W') 
            if self.data=='TREC':
                W = tf.Variable(tf.zeros(shape = [self.represent.shape[-1],6]),name = 'W')
                b = tf.Variable(tf.zeros([6]), name = 'b')
            elif self.data=='ag_news':
                W = tf.Variable(tf.zeros(shape = [self.represent.shape[-1],4]),name = 'W')
                b = tf.Variable(tf.zeros([4]), name = 'b')
            else:
                W = tf.Variable(tf.zeros(shape = [self.represent.shape[-1],2]),name = 'W')
                b = tf.Variable(tf.zeros([2]), name = 'b')


            #W = tf.Variable(tf.zeros(shape = [self.represent.shape[-1],6]),name = 'W')
            # W = tf.Variable(tf.zeros(shape = [(self.embedding_size - self.filter_sizes[0] + 1) * self.num_filters * 2, 2]),name = 'W')
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
            #b = tf.Variable(tf.zeros([6]), name = 'b')
            self.para.append(W)
            self.para.append(b)
            # print(self.represent) # [batch, 7680]
            # print(self.match_represent) # [batch, 51]
            # self.represent = tf.concat([self.represent, self.M_qa_2], -1)
            # self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "scores")
            # self.tmp = tf.concat([tf.expand_dims(self.represent, 1),tf.expand_dims(self.represent2, 1)] , 1)
            # self.tmp = tf.multiply(self.tmp, self.sd_weight) 
            # self.tmp = tf.reduce_sum(self.tmp, 1)
            # self.qa = tf.multiply(self.represent_q, self.represent_a)
            # self.represent = tf.concat([self.represent, self.q_trace, self.a_trace], -1)
            # self.represent, self.trace_q, self.trace_a
            # self.represent = tf.concat([self.represent_q, self.represent_a, self.q_all, self.a_all], -1)
            self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "scores")
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
            
    def create_loss(self):
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            print(self.input_y)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            self.loss = tf.reduce_mean(losses)   # + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def concat_embedding(self,words_indice,lens):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)

        tmp = tf.zeros(shape=[words_indice.get_shape()[1], self.embedding_size])

        for i in range(self.batch_size):

            t_w = embedded_chars_q[i,:lens[i],:]

            t_c = tmp[lens[i]:,:]

            if i == 0:

                w_t = tf.expand_dims(tf.concat([t_w, t_c], 0), 0)

            else:

                w_t = tf.concat([w_t, tf.expand_dims(tf.concat([t_w, t_c], 0), 0)], 0)

        return tf.expand_dims(w_t,-1)

        # position_embedding = tf.nn.embedding_lookup(self.position_W,position_indice)
        # overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        # return tf.expand_dims(embedded_chars_q,-1)

        
                
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
        self.qa_0, self.qa_1, self.qa_2 = self.narrow_convolution(tf.expand_dims(self.embedded_chars_q,-1))
        # self.a = self.narrow_convolution(tf.expand_dims(self.density_a,-1))
        # self.qa = self.narrow_convolution(tf.expand_dims(self.embedded_chars_q,-1))

    def pooling_graph(self):
        with tf.name_scope('pooling'):
      
            # pooling = self.max_pooling(self.qa)
            # print self.pooling
            # self.represent = tf.reshape(pooling,[-1,self.num_filters * len(self.filter_sizes)])
            # raw_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.a,1))

            # col_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.a,2))
            
            # self.represent_a = tf.concat([raw_pooling,col_pooling],1)

            # raw_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qa,1))

            # col_pooling = tf.contrib.layers.flatten(tf.reduce_max(self.qa,2))

            # raw_pooling = tf.contrib.layers.flatten(tf.reduce_mean(self.qa,1))

            # col_pooling = tf.contrib.layers.flatten(tf.reduce_mean(self.qa,2))
            
            # self.represent = tf.concat([raw_pooling,col_pooling],1)

            pooling_0 = self.max_pooling(self.qa_0)
            pooling_1 = self.max_pooling(self.qa_1)
            pooling_2 = self.max_pooling(self.qa_2)

            pooling_0 = tf.reshape(pooling_0, [self.batch_size, -1])
            pooling_1 = tf.reshape(pooling_1, [self.batch_size, -1])
            pooling_2 = tf.reshape(pooling_2, [self.batch_size, -1])

            self.represent = tf.concat([pooling_0,pooling_1,pooling_2],1)
            # self.represent = pooling_0

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
            h = tf.nn.tanh(tf.nn.bias_add(conv, self.kernels[i][1]), name="tanh-1")
            cnn_outputs.append(h)
        # cnn_reshaped = tf.concat(cnn_outputs,3)
        # return cnn_reshaped
        return cnn_outputs[0], cnn_outputs[1], cnn_outputs[2]

    

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
        # pooled = tf.nn.max_pool(
        #             conv,
        #             # ksize = [1, self.total_embedding_dim, self.total_embedding_dim, 1],
        #             ksize = [1, 5, 5, 1],
        #             strides = [1, 1, 1, 1],
        #             padding = 'VALID',
        #             name = "pool")
        pooled = tf.nn.max_pool(
                    conv,
                    # ksize = [1, self.total_embedding_dim, self.total_embedding_dim, 1],
                    ksize = [1, 5, 5, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
        return pooled

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

    def trace_represent(self, matrix):
        # self.density_diag = tf.matrix_diag_part(self.M_qa)
        # self.density_trace = tf.expand_dims(tf.trace(self.M_qa),-1)
        # self.match_represent = tf.concat([self.density_diag,self.density_trace],1)
        # self.density_diag = tf.matrix_diag_part(self.density_a)
        # self.density_trace = tf.expand_dims(tf.trace(self.density_a),-1)
        # self.match_represent = tf.concat([self.density_diag,self.density_trace],1)
        density_diag = tf.matrix_diag_part(matrix)
        density_trace = tf.expand_dims(tf.trace(matrix),-1)
        match_represent = tf.concat([density_diag,density_trace], 1)

        return match_represent
    def build_graph(self):
        self.create_placeholder()
        self.density_weighted()
        self.add_embeddings()
        self.convolution()
        self.pooling_graph()
        self.feed_neural_work()
        self.create_loss()

if __name__ == '__main__':
    cnn = QA_quantum(max_input_left = 33,
                max_input_right = 40,
                vocab_size = 5000,
                embedding_size = 50,
                batch_size = 3,
                embeddings = None,
                dropout_keep_prob = 1,
                filter_sizes = [40],
                num_filters = 65,
                l2_reg_lambda = 0.0,
                is_Embedding_Needed = False,
                trainable = True,
                overlap_needed = False,
                pooling = 'max',
                position_needed = False)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_y = np.ones((3,2))

    input_overlap_q = np.ones((3,33))
    input_overlap_a = np.ones((3,40))
    q_posi = np.ones((3,33))
    a_posi = np.ones((3,40))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.input_y:input_y,
            cnn.q_overlap:input_overlap_q,
            cnn.a_overlap:input_overlap_a,
            cnn.q_position:q_posi,
            cnn.a_position:a_posi
        }
       
        see,question,answer,scores = sess.run([cnn.embedded_chars_q,cnn.question,cnn.answer,cnn.scores],feed_dict)
        # print see

       
