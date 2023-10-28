import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from Hypers import *

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

def pairwise_cosine_similarity(x,y,zero_diagonal):
    x_norm = tf.math.l2_normalize(x)  #(?,M,d)
    y_norm = tf.math.l2_normalize(y)  #(?,N,d)
    distance = tf.matmul(x_norm,y_norm,transpose_b=True)
    if zero_diagonal:
        mask = tf.tile(tf.expand_dims(tf.eye(get_shape(x)[1]),0),[get_shape(x)[0],1,1])
        distance = distance*(1-mask)
        
    return distance

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

def AttentivePooling(dim1, dim2):
    vecs_input = Input(shape=(dim1, dim2), dtype='float32')
    user_vecs = Dropout(0.2)(vecs_input)
    user_att = Dense(200, activation='tanh')(user_vecs)
    user_att = layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1, 1))([user_vecs, user_att])
    model = Model(vecs_input, user_vec)
    return model


def get_doc_encoder(title_word_embedding_matrix):
    news_input = Input(shape=(TITLE_SIZE,), dtype='int32')

    title_word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], 
                                           title_word_embedding_matrix.shape[1],
                                           weights=[title_word_embedding_matrix], trainable=True)
    word_vecs = title_word_embedding_layer(news_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    word_rep = Attention(20, 20)([droped_vecs] * 3)
    droped_rep = Dropout(0.2)(word_rep)
    title_vec = AttentivePooling(30, 400)(droped_rep)

    sentEncodert = Model(news_input, title_vec)
    return sentEncodert


class PolyAttention(Layer):
    def __init__(self, hidden_dim, num_context, **kwargs):
        self.hidden_dim = hidden_dim
        self.num_context=num_context
        super(PolyAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        trainable = True
        self.context = self.add_weight(name='c',
                                  shape=(self.num_context,self.hidden_dim),
                                  initializer=keras.initializers.glorot_uniform(seed=0),
                                  trainable=trainable,
                                  regularizer = keras.regularizers.l2(0.0001))
        
        self.projection = layers.Dense(self.hidden_dim,activation='tanh',use_bias=False)  
        
        self.lamada = self.add_weight(name='w',
                                  shape=(1,),
                                  initializer=tensorflow.keras.initializers.Constant(value=0),
                                  trainable=trainable)

        super(PolyAttention, self).build(input_shape)

    def call(self, inputs):
        x, category_bias,att_mask = inputs       
        proj = self.projection(x)               
        weights = tf.matmul(proj, self.context, transpose_b=True) + self.lamada*category_bias

        weights = tf.transpose(weights,[0,2,1]) 
        masks = tf.expand_dims(att_mask,1) 
        
        weights = weights - (1 - masks) * 1e12
        weights = tf.nn.softmax(weights, axis=-1) 
        
        poly_repr = tf.matmul(weights,x) 
                
        return poly_repr

    def compute_output_shape(self, input_shape):
        return input_shape[0][0],self.num_context,input_shape[0][2]


class ComputeMasking(layers.Layer):
    """Compute if inputs contains zero value.

    Returns:
        bool tensor: True for values not equal to zero.
    """

    def __init__(self, **kwargs):
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = K.not_equal(inputs, 0)
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):   
        config = super().get_config().copy()
        return config

class Target_weight(Layer):
    def __init__(self, hidden, **kwargs):
        self.hidden = hidden
        super(Target_weight, self).__init__(**kwargs)

    def build(self, input_shape):
        self.linear = layers.Dense(self.hidden,activation='tanh',use_bias=False)
        
        super(Target_weight, self).build(input_shape)

    def call(self,x):
        query, key, value = x
        proj = self.linear(query) 
        att = tf.matmul(proj, key) 
        att = tf.squeeze(att,-1) 
        att = tf.nn.softmax(att,-1) 
        scores = tf.reduce_sum(tf.multiply(att,value),-1,keep_dims=True) 
           
        return scores

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)

class Inner_model(Layer):
    def __init__(self,news_encoder, vert_embedding_matrix, num_context, **kwargs):
        self.vert_encoder = Embedding(vert_embedding_matrix.shape[0], 
                               vert_embedding_matrix.shape[1],
                               weights=[vert_embedding_matrix], trainable=True)
        self.num_context = num_context
        self.poly_attn = PolyAttention(200,num_context)
        self.news_encoder = news_encoder
        self.vert_dropout = layers.Dropout(0.2)
        self.Target_weight = Target_weight(400)

        super(Inner_model, self).__init__(**kwargs)

    def build(self, input_shape):
        trainable = False
        
        self.w = self.add_weight(name='w',
                                  shape=(1,),
                                  initializer=tensorflow.keras.initializers.Constant(value=0),
                                  trainable=trainable)

        super(Inner_model, self).build(input_shape)

    def get_multiple_user_pre(self,user_clicked_title, user_clicked_vert, candidate_vert,user_mask):
        his_title_emd = TimeDistributed(self.news_encoder)(user_clicked_title)
        
        his_vert_emd = self.vert_encoder(user_clicked_vert)
        his_vert_emd = self.vert_dropout(his_vert_emd)
        his_vert_emd = tf.math.l2_normalize(his_vert_emd, -1) 
        
        can_vert_emd = self.vert_encoder(candidate_vert)
        can_vert_emd = self.vert_dropout(can_vert_emd)
        can_vert_emd = tf.math.l2_normalize(can_vert_emd, -1) 
        can_vert_emd = tf.expand_dims(can_vert_emd,1) 
        
        vert_cosine = tf.matmul(his_vert_emd,can_vert_emd,transpose_b=True) 
        
        print(his_title_emd,vert_cosine,user_mask)
        mul_user_interests = self.poly_attn([his_title_emd,vert_cosine,user_mask])
        return mul_user_interests
        
    def call(self, x):
        
        user_clicked_title, candidate_title, user_clicked_vert, candidate_vert, user_mask = x
        mul_user_interests = self.get_multiple_user_pre(user_clicked_title, user_clicked_vert, candidate_vert, user_mask)
        
        can_emd = self.news_encoder(candidate_title)       
        can_emd = tf.expand_dims(can_emd,-1) 
        scores = tf.matmul(mul_user_interests,can_emd) 
        scores = tf.squeeze(scores,axis=-1) 
               
        final_scores = self.Target_weight([mul_user_interests,can_emd,scores])
        
        
        return final_scores,mul_user_interests
    

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1),(input_shape[0][0],self.num_context,input_shape[0][2])

def get_miner(news_encoder,vert_embedding_matrix):
    inner_model = Inner_model(news_encoder,vert_embedding_matrix,32)
    
    clicked_title_input = Input(shape=(MAX_CLICK,TITLE_SIZE+1,), dtype='int32')    
    title_inputs = Input(shape=(TITLE_SIZE+1,),dtype='int32') 

        
    u_input = keras.layers.Lambda(lambda x:x[:,:,:TITLE_SIZE])(clicked_title_input)  # (?,50,30)
    c_input = keras.layers.Lambda(lambda x:x[:,:TITLE_SIZE])(title_inputs)  # (?,30)
    
    
    clicked_news_cat_input = keras.layers.Lambda(lambda x:x[:,:,TITLE_SIZE])(clicked_title_input)
    title_cat_input = keras.layers.Lambda(lambda x:x[:,TITLE_SIZE])(title_inputs)
    
    user_mask = ComputeMasking()(tf.reduce_sum(u_input,-1))
    
    [scores,mul_user_pre] = inner_model([u_input,c_input,clicked_news_cat_input,title_cat_input,user_mask])
    
    model = Model([title_inputs, clicked_title_input], [scores,mul_user_pre])
    model1 = Model([title_inputs, clicked_title_input], scores)
    
    return model,model1


def create_model(title_word_embedding_matrix,vert_embedding_matrix):
    
    news_encoder = get_doc_encoder(title_word_embedding_matrix)
    
    miner,inner_model = get_miner(news_encoder,vert_embedding_matrix)

    clicked_title_input = Input(shape=(MAX_CLICK,MAX_TITLE+1,), dtype='int32')    
    title_inputs = Input(shape=(1+npratio,MAX_TITLE+1,),dtype='int32') 
    
    
    scores = []
    user_loss = []
    for i in range(1+npratio):
        news_vec = keras.layers.Lambda(lambda x:x[:,i,:])(title_inputs)
        score,u_pre = miner([news_vec,clicked_title_input])
        u_loss = pairwise_cosine_similarity(u_pre,u_pre,True) 
        u_loss = tf.reduce_mean(u_loss)
        user_loss.append(u_loss)
        scores.append(score)
        
    scores = keras.layers.Concatenate(axis=-1)(scores)
    
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)   
    user_loss = tf.reduce_mean(user_loss)
    
    model = Model([title_inputs, clicked_title_input],logits) 
    model.add_loss(0.8*user_loss)
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=Adam(lr=0.00005), 
                  metrics=['acc'])

    return model,inner_model