import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from Hypers import *

class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.int_shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.int_shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.int_shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.int_shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.int_shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    user_vecs =Dropout(0.2)(vecs_input)
    user_att = Dense(200,activation='tanh')(user_vecs)
    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model

class MCNN(Layer):
 
    def __init__(self, dim, topic_num, **kwargs):
        self.dim = dim
        self.topic=topic_num
        self.k=3
        self.pad_len = int((self.k-1)/2)
        self.Den = keras.layers.Dense(self.k*self.dim,activation='relu')
        self.pad_layer = keras.layers.ZeroPadding1D(padding=self.pad_len)
        
        super(MCNN, self).__init__(**kwargs)
 
    def build(self, input_shape):       
        self.C = self.add_weight(name='C',
                                  shape=(self.topic,self.dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
        super(MCNN, self).build(input_shape)
 
    def call(self, x):
        kc = self.Den(self.C)
        att = K.dot(x,K.transpose(self.C))         
        output = []
        
        x_pad=self.pad_layer(x)
               
        for i in range(self.pad_len,K.int_shape(x)[1]+self.pad_len):
            l=i-self.pad_len
            r=i+self.pad_len+1
            ki = K.dot(att[:,i-self.pad_len],kc) 
            ki = K.reshape(ki,shape=(-1,self.k,self.dim))
            output.append(tf.reduce_sum(x_pad[:,l:r,:]*ki,axis=-2))
        output = tf.stack(output,axis=1)
            
        return output

    def compute_output_shape(self, input_shape):
        return input_shape
    
    
def get_doc_encoder(title_word_embedding_matrix):
    title_input = Input(shape=(MAX_TITLE,), dtype='int32')
    
    title_word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], title_word_embedding_matrix.shape[1], weights=[title_word_embedding_matrix],trainable=True)
    word_vecs = title_word_embedding_layer(title_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    word_rep = Attention(15,20)([droped_vecs]*3)
    droped_rep = Dropout(0.2)(word_rep)
    MCNN_layer = MCNN(300,5)
    cnn_rep = MCNN_layer(droped_rep)
    
    title_vec = keras.layers.Add()([droped_rep,cnn_rep])
    title_vec = Dropout(0.2)(title_vec)
    print(droped_rep,cnn_rep,title_vec)
    
    title_vec = AttentivePooling(MAX_TITLE,300)(title_vec)
            
    sentEncodert = Model(title_input, title_vec)
    return sentEncodert,MCNN_layer

def DenseAtt():
    vec_input = Input(shape=(400*2,),dtype='float32')
    vec = Dense(400,activation='tanh')(vec_input)
    vec = Dense(256,activation='tanh')(vec)
    score = Dense(1)(vec)
    return Model(vec_input,score)

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

def get_Topic_Loss(C):
    C_norm = tf.math.l2_normalize(C, axis=-1)
    C_sim = K.dot(C_norm,K.transpose(C_norm)) #(6,6)
    C_sim /= 0.1 
    c_label = tf.eye(K.int_shape(C_norm)[0])
    L_m = tf.nn.softmax_cross_entropy_with_logits(labels=c_label, logits= C_sim) 
    L_m = tf.reduce_mean(L_m)
    return L_m

def user_contrastive_loss(user,aug_user):
    user_norm = tf.math.l2_normalize(user, axis=-1)
    aug_user_norm = tf.math.l2_normalize(aug_user, axis=-1)
    C_sim = K.dot(user_norm,K.transpose(aug_user_norm)) #(6,6)
    C_sim /= 0.1 
    print(user_norm)
    c_label = tf.eye(get_shape(user_norm)[0])
    L_m = tf.nn.softmax_cross_entropy_with_logits(labels=c_label, logits= C_sim) 
    L_m = tf.reduce_mean(L_m)
    return L_m

def get_user_encoder(news_encoder):
    clicked_title_input =  Input(shape=(MAX_CLICK,MAX_TITLE,), dtype='float32')
    clicked_news_vecs = TimeDistributed(news_encoder)(clicked_title_input)
    clicked_news_vecs = Dropout(0.2)(clicked_news_vecs)
    print(clicked_news_vecs)
    user_vec = AttentivePooling(MAX_CLICK,300)(clicked_news_vecs) #(?,400)
    model = Model(clicked_title_input,user_vec)
    return model 
   
def MCCM(title_word_embedding_matrix):        
    
    clicked_title_input =  Input(shape=(MAX_CLICK,MAX_TITLE,), dtype='float32') 
    title_inputs = Input(shape=(1+npratio,MAX_TITLE,),dtype='float32')
    aug_clicked_title_input =  Input(shape=(MAX_CLICK,MAX_TITLE,), dtype='float32') 
    
    news_encoder,MCNN_layer = get_doc_encoder(title_word_embedding_matrix)    
    news_encoder.compute_output_shape = lambda x : (x[0],300)

    
    
    user_encoder =  get_user_encoder(news_encoder)
    
    user_vec = user_encoder(clicked_title_input)
    aug_user_vec = user_encoder(aug_clicked_title_input)
    
    title_vecs = TimeDistributed(news_encoder)(title_inputs) #(?,5,400)
        
    scores = keras.layers.Dot(axes=-1)([title_vecs,user_vec])
    
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)     

    model = Model([title_inputs, clicked_title_input, aug_clicked_title_input],logits) # max prob_click_positive
    
    C = MCNN_layer.C  #(c,dim)
    L_m = get_Topic_Loss(C)
    
    model.add_loss(0.1*L_m)
    
    L_user = user_contrastive_loss(user_vec,aug_user_vec)
    model.add_loss(0.1*L_user)
    
        
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=Adam(lr=0.00005), 
                  metrics=['acc'])

    return model,news_encoder,user_encoder