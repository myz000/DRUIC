import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from base_model import BaseModel
from layers import AttLayer, SelfAttention, ComputeMasking, OverwriteMasking

__all__ = ["DRUICModel"]


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape


class Purpose_Router_layer(layers.Layer):
    def __init__(self, input_dim, output_dim, e, **kwargs):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.t = e
        self.seed = 0
        super(Purpose_Router_layer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.PRW = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=self.seed),
            trainable=True,
            regularizer=keras.regularizers.l2(0.0001)
        )

        super(Purpose_Router_layer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        topic_norm = self.PRW
        inputs = tf.math.l2_normalize(inputs, axis=-1)
        topic_norm = tf.math.l2_normalize(self.PRW, axis=0)
        weights = tf.matmul(inputs, topic_norm)
        weights_ori = weights/self.t

        weights = tf.linalg.diag_part(weights, k=-2)
        weights = (weights+1.0)/2.0

        if mask != None:
            weights = weights*mask

        return weights, weights_ori

    def compute_mask(self, input, input_mask=None):
        return None

    def get_config(self):
        config = super().get_config().copy()
        return config


class DRUIC(BaseModel):
    def __init__(
        self, hparams, iterator_creator, seed=None
    ):
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.channels = hparams.channels
        self.PR_tp = hparams.purpose_router_tp
        self.hidden_dim = hparams.head_dim + hparams.filter_num
        self.seed = 0
        self.infer_loss_weight = hparams.infer_loss_weight

        super().__init__(
            hparams, iterator_creator, seed=seed
        )

    def _get_all_training_news(self):
        self.train_iterator.init_news(news_file)

    def _get_train_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["user_index_batch"],
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
            batch_data["clicked_title_pos_batch"],
            batch_data["augmented_clicked_title_batch"],
            batch_data["users_augmented_mask_batch"]
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_test_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_graph(self):
        hparams = self.hparams
        print("start _build_graph...")
        model, scorer = self._build_DRUIC()
        return model, scorer

    def _news_infer_loss(self, x, PR1, PR2, mask=None):

        hparams = self.hparams
        label = tf.tile(tf.eye(self.channels), [get_shape(x)[0], 1])

        x1 = x[:, :, :hparams.head_dim]
        x2 = x[:, :, hparams.head_dim:]

        _, x_class1 = PR1(x1, mask)
        _, x_class2 = PR2(x2, mask)

        x_class1 = tf.reshape(x_class1, [-1, self.channels])
        x_class2 = tf.reshape(x_class2, [-1, self.channels])

        infer_loss1 = tf.nn.softmax_cross_entropy_with_logits(
            labels=label, logits=x_class1)
        infer_loss1 = tf.reduce_mean(infer_loss1)

        infer_loss2 = tf.nn.softmax_cross_entropy_with_logits(
            labels=label, logits=x_class2)
        infer_loss2 = tf.reduce_mean(infer_loss2)

        infer_loss = infer_loss1+infer_loss2

        return infer_loss

    def get_projection_layer(self, hidden_dim):

        proj_layer = tf.keras.models.Sequential()

        proj_layer.add(layers.Dense(hidden_dim,
                                    use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(0.0001)))

        proj_layer.add(layers.BatchNormalization())
        proj_layer.add(layers.ReLU())
        proj_layer.add(layers.Dense(hidden_dim,
                                    use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(0.0001)))

        proj_layer.add(layers.BatchNormalization())

        return proj_layer

    def get_prediction_layer(self, hidden_dim):

        proj_layer = tf.keras.models.Sequential()
        proj_layer.add(layers.Dense(int(hidden_dim/4),
                                    use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(0.0001)))

        proj_layer.add(layers.ReLU())
        proj_layer.add(layers.BatchNormalization())
        proj_layer.add(layers.Dense(hidden_dim))
        return proj_layer

    def compute_loss(self, p, z):
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=-1)
        z = tf.math.l2_normalize(z, axis=-1)
        return -tf.reduce_sum(tf.math.multiply(p, z), axis=-1)

    def _news_contrastive_loss(self, input1, input2, his_mask, Predict_layer, Project_layer):
        hparams = self.hparams

        z1, z2 = Project_layer(input1), Project_layer(input2)
        p1, p2 = Predict_layer(z1), Predict_layer(z2)
        loss = self.compute_loss(p1, z2) / 2 + self.compute_loss(p2, z1) / 2
        loss = tf.reduce_sum(tf.math.multiply(
            loss, his_mask))/tf.reduce_sum(his_mask)

        return loss

    def add_contrastive_loss(self, input1, input2, mask, hidden_norm=True, temperature=0.07, weights=1.0):
        hparams = self.hparams

        hidden1 = tf.convert_to_tensor(input1)
        hidden2 = tf.convert_to_tensor(input2)

        if hidden_norm:
            hidden1 = tf.math.l2_normalize(hidden1, -1)
            hidden2 = tf.math.l2_normalize(hidden2, -1)

        user_num = get_shape(hidden1)[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(user_num), user_num * 2)
        masks = tf.one_hot(tf.range(user_num), user_num)

        LARGE_NUM = 1e9

        logits_aa = tf.matmul(hidden1, hidden1_large,
                              transpose_b=True) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2_large,
                              transpose_b=True) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2_large,
                              transpose_b=True) / temperature
        logits_ba = tf.matmul(hidden2, hidden1_large,
                              transpose_b=True) / temperature

        loss_a = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ba, logits_bb], 1))
        loss = loss_a + loss_b
        loss = tf.reduce_sum(tf.math.multiply(loss, mask))/tf.reduce_sum(mask)
        return loss

    def _build_userencoder(self, newsencoder):
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        masks = ComputeMasking()(tf.reduce_sum(his_input_title, -1, keepdims=True))

        click_title_presents = layers.TimeDistributed(
            newsencoder)(his_input_title)

        masks = masks[:, :, 0]

        Self_Att_Layer = [
            SelfAttention(1, self.hidden_dim, seed=self.seed)
            for i in range(self.channels)
        ]
        Att_layers = [
            AttLayer(hparams.attention_hidden_dim,
                     r=hparams.ssa_r, seed=self.seed)
            for i in range(self.channels)
        ]

        y = [
            Self_Att_Layer[i]([click_title_presents[:, :, i, :]] * 3)
            for i in range(self.channels)
        ]

        user_present = [
            Att_layers[i](y[i], mask=masks)
            for i in range(self.channels)
        ]

        user_present = tf.stack(user_present)
        user_present = tf.transpose(user_present, perm=[1, 0, 2])
        model = keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):

        hparams = self.hparams
        sequences_input_title = keras.Input(
            shape=(hparams.title_size,), dtype="int32")

        masks = ComputeMasking()(sequences_input_title)

        embedded_sequences_title = embedding_layer(sequences_input_title)

        y0 = layers.Dropout(hparams.dropout)(embedded_sequences_title)

        y = SelfAttention(hparams.channels, hparams.head_dim,
                          seed=self.seed)([y0, y0, y0])
        y = layers.Dropout(hparams.dropout)(y)
        y = layers.Reshape(
            (hparams.title_size, self.channels, hparams.head_dim),)(y)
        MultiAttLayers = [
            AttLayer(hparams.attention_hidden_dim,
                     r=hparams.ssa_r, seed=self.seed)
            for i in range(hparams.channels)
        ]
        pred_title = [
            MultiAttLayers[i](y[:, :, i, :], mask=masks)
            for i in range(hparams.channels)
        ] 
        pred_title = layers.Concatenate(axis=1)(pred_title) 
        pred_title = layers.Reshape(
            (self.channels, hparams.head_dim),)(pred_title)  

        y1 = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(
                seed=self.seed),
            kernel_regularizer=keras.regularizers.l2(0.0001)
        )(y0)
        y1 = layers.Dropout(hparams.dropout)(y1)
        y1 = layers.Masking()(
            OverwriteMasking()([y1, ComputeMasking()(sequences_input_title)])
        )

        MultiAttLayers_1 = [AttLayer(
            hparams.attention_hidden_dim, r=hparams.ssa_r, seed=self.seed) for i in range(hparams.channels)]
        pred_title_1 = [MultiAttLayers_1[i](
            y1, mask=masks) for i in range(hparams.channels)]  
        pred_title_1 = layers.Concatenate(axis=1)(pred_title_1)  
        pred_title_1 = layers.Reshape(
            (self.channels, hparams.filter_num),)(pred_title_1)  

        pred_title_final = layers.Concatenate()(
            [pred_title, pred_title_1]) 
        
        model = keras.Model(sequences_input_title,
                            pred_title_final, name="news_encoder")

        return model

    def _build_DRUIC(self):
        hparams = self.hparams

        user_indexes = keras.Input(shape=(1,), dtype="int32")

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )

        pred_input_title_one = keras.Input(
            shape=(hparams.title_size,), dtype="int32"
        )

        his_input_pos_title = keras.Input(
            shape=(hparams.his_size, 2, hparams.title_size), dtype="int32"
        )

        users_augmented_input = keras.Input(
            shape=(2, hparams.his_size, hparams.title_size), dtype="int32"
        )
        users_augmented_mask = keras.Input(shape=(1,), dtype="float32")

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        PurposeRouter_layer = Purpose_Router_layer(
            hparams.head_dim, self.channels, self.PR_tp, name="Purpose_Router_layer")
        self.PurposeRouter = PurposeRouter_layer

        PurposeRouter_layer_1 = Purpose_Router_layer(
            hparams.filter_num, self.channels, self.PR_tp, name="Purpose_Router_layer_1")
        self.PurposeRouter_1 = PurposeRouter_layer_1

        titleencoder = self._build_newsencoder(
            embedding_layer)
        titleencoder.compute_output_shape = lambda x: (
            x[0], self.channels, self.hidden_dim)
        
        self.newsencoder = titleencoder
              
        self.userencoder = self._build_userencoder(
            titleencoder)
        self.userencoder.compute_output_shape = lambda x: (
            x[0], self.channels, self.hidden_dim)

        user_present = self.userencoder(his_input_title)  

        news_present = layers.TimeDistributed(self.newsencoder)(
            pred_input_title)  

        news_present_one = self.newsencoder(pred_input_title_one) 
       
        preds = tf.reduce_sum(tf.multiply(news_present, tf.tile(
            tf.expand_dims(user_present, 1), [1, 5, 1, 1])), -1)  
        preds = tf.reduce_sum(preds, -1)
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = tf.reduce_sum(tf.multiply(
            news_present_one, user_present), -1)
        pred_one = tf.reduce_sum(pred_one, -1, keepdims=True)
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([user_indexes, his_input_title, pred_input_title,
                            his_input_pos_title, users_augmented_input, users_augmented_mask], preds)
        
        scorer = keras.Model([his_input_title, pred_input_title_one], pred_one)
       

        # The disentangled loss
        infer_input_title = tf.reshape(
            his_input_title, [-1, hparams.title_size])
        infer_input_title = self.tf_unique_2d(infer_input_title)
        his_news_present = self.newsencoder(infer_input_title)  
        infer_loss = self._news_infer_loss(
            his_news_present, self.PurposeRouter, self.PurposeRouter_1)

        if self.infer_loss_weight > 0.0:
            model.add_loss(self.infer_loss_weight*infer_loss)
            
        model.add_metric(self.infer_loss_weight*infer_loss,
                         name='infer_loss', aggregation='mean')

        # News-level contrastive loss
        if hparams.ctr1_loss_weight > 0.0:
            ctr1_loss = 0

            his_pos_emd_1 = layers.TimeDistributed(self.newsencoder)(
                his_input_pos_title[:, :, 0, :])  
            his_pos_emd_2 = layers.TimeDistributed(self.newsencoder)(
                his_input_pos_title[:, :, 1, :]) 
            his_pos_emd_11 = tf.reshape(
                his_pos_emd_1[:, :, :, :hparams.head_dim], [-1, self.channels*hparams.head_dim])
            his_pos_emd_21 = tf.reshape(
                his_pos_emd_2[:, :, :, :hparams.head_dim], [-1, self.channels*hparams.head_dim])

            his_pos_emd_12 = tf.reshape(
                his_pos_emd_1[:, :, :, hparams.head_dim:], [-1, self.channels*hparams.filter_num])
            his_pos_emd_22 = tf.reshape(
                his_pos_emd_2[:, :, :, hparams.head_dim:], [-1, self.channels*hparams.filter_num])

            self.ctr1_Project_layer_1 = self.get_projection_layer(
                self.channels*hparams.head_dim)
            self.ctr1_Project_layer_2 = self.get_projection_layer(
                self.channels*hparams.filter_num)
            self.ctr1_Prediction_layer_1 = self.get_prediction_layer(
                self.channels*hparams.head_dim)
            self.ctr1_Prediction_layer_2 = self.get_prediction_layer(
                self.channels*hparams.filter_num)

            his_mask = ComputeMasking()(tf.reduce_sum(his_input_title, -1))  

            ctr1_loss += self._news_contrastive_loss(his_pos_emd_11, his_pos_emd_21, tf.reshape(
                his_mask, [-1]), self.ctr1_Prediction_layer_1, self.ctr1_Project_layer_1)
            ctr1_loss += self._news_contrastive_loss(his_pos_emd_12, his_pos_emd_22, tf.reshape(
                his_mask, [-1]), self.ctr1_Prediction_layer_2, self.ctr1_Project_layer_2)

            model.add_loss(hparams.ctr1_loss_weight*ctr1_loss)
        else:
            ctr1_loss = tf.zeros_like(infer_loss)
        model.add_metric(hparams.ctr1_loss_weight*ctr1_loss,
                         name='ctr1_loss', aggregation='mean')


        # User-level contrastive loss
        augmented_reps_1 = self.userencoder(
            users_augmented_input[:, 0, :, :])  
        augmented_reps_2 = self.userencoder(
            users_augmented_input[:, 1, :, :]) 

        augmented_reps_1 = tf.reshape(
            augmented_reps_1, [-1, self.channels*self.hidden_dim])
        augmented_reps_2 = tf.reshape(
            augmented_reps_2, [-1, self.channels*self.hidden_dim])

        ctr2_loss = self.add_contrastive_loss(
            augmented_reps_1, augmented_reps_2, users_augmented_mask)

        if hparams.ctr2_loss_weight > 0.0:
            model.add_loss(hparams.ctr2_loss_weight*ctr2_loss)
        model.add_metric(hparams.ctr2_loss_weight*ctr2_loss,
                         name='ctr2_loss', aggregation='mean')

        return model, scorer

    def get_config(self):
        config = super().get_config().copy()
        return config

    def tf_unique_2d(self, x):
        x_shape = tf.shape(x)  
        x1 = tf.tile(x, [1, x_shape[0]])
        x2 = tf.tile(x, [x_shape[0], 1])

        x1_2 = tf.reshape(x1, [x_shape[0] * x_shape[0], x_shape[1]])
        x2_2 = tf.reshape(x2, [x_shape[0] * x_shape[0], x_shape[1]])
        cond = tf.reduce_all(tf.equal(x1_2, x2_2), axis=1)
        # reshaping cond to match x1_2 & x2_2
        cond = tf.reshape(cond, [x_shape[0], x_shape[0]])
        cond_shape = tf.shape(cond)
        # convertin condition boolean to int
        cond_cast = tf.cast(cond, tf.int32)
        # replicating condition tensor into all 0's
        cond_zeros = tf.zeros(cond_shape, tf.int32)

        # CREATING RANGE TENSOR
        r = tf.range(x_shape[0])
        r = tf.add(tf.tile(r, [x_shape[0]]), 1)
        r = tf.reshape(r, [x_shape[0], x_shape[0]])

        # converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
        f1 = tf.multiply(tf.ones(cond_shape, tf.int32), x_shape[0] + 1)
        f2 = tf.ones(cond_shape, tf.int32)
        # if false make it max_index+1 else keep it 1
        cond_cast2 = tf.where(tf.equal(cond_cast, cond_zeros), f1, f2)

        # multiply range with new int boolean mask
        r_cond_mul = tf.multiply(r, cond_cast2)
        r_cond_mul2 = tf.reduce_min(r_cond_mul, axis=1)
        r_cond_mul3, unique_idx = tf.unique(r_cond_mul2)
        r_cond_mul4 = tf.subtract(r_cond_mul3, 1)

        # get actual values from unique indexes
        op = tf.gather(x, r_cond_mul4)

        return (op)
