from os.path import join
import abc
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from newsrec_utils import cal_metric


__all__ = ["BaseModel"]


class BaseModel:
    """Basic class of models

    Attributes:
        hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
        iterator_creator_train (obj): An iterator to load the data in training steps.
        iterator_creator_train (obj): An iterator to load the data in testing steps.
        graph (obj): An optional graph.
        seed (int): Random seed.
    """

    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
    ):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator_train (obj): An iterator to load the data in training steps.
            iterator_creator_train (obj): An iterator to load the data in testing steps.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """       
        
        self.seed = seed
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        self.train_iterator = iterator_creator(
            hparams,
            hparams.npratio,
            col_spliter="\t",
        )
        
        self.valid_iterator = iterator_creator(
            hparams,
            col_spliter="\t",
        )

        self.test_iterator = iterator_creator(
            hparams,
            col_spliter="\t",
        )

        self.hparams = hparams
        self.support_quick_scoring = hparams.support_quick_scoring

        # set GPU use with on demand growth
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )

        # set this TensorFlow session as the default session for Keras
        tf.compat.v1.keras.backend.set_session(sess)

        # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
        # Otherwise, their weights will be unavailable in the threads after the session there has been set
        self.model, self.scorer = self._build_graph()

        self.loss = self._get_loss()
        self.train_optimizer = self._get_opt()

        self.model.compile(loss=self.loss, optimizer=self.train_optimizer)

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained glove embeddings file path.

        Returns:
            np.array: A constant numpy array.
        """

        return np.load(file_path)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    @abc.abstractmethod
    def _get_input_label_from_iter(self, batch_data):
        """Subclass will implement this"""
        pass

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss

        Returns:
            obj: Loss function or loss function name
        """
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif self.hparams.loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _get_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            obj: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adam":
            train_opt = keras.optimizers.Adam(lr=lr)

        return train_opt

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.

        Args:
            logit (obj): Base prediction value.
            task (str): A task (values: regression/classification)

        Returns:
            obj: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

        
    def train(self, train_batch_data):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        train_input, train_label = self._get_train_input_label_from_iter(train_batch_data)
        rslt = self.model.train_on_batch(train_input, train_label)
        return rslt

    def eval(self, eval_batch_data):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        eval_input, eval_label = self._get_test_input_label_from_iter(eval_batch_data)
        imp_index = eval_batch_data["impression_index_batch"]

        pred_rslt = self.scorer.predict_on_batch(eval_input)

        return pred_rslt, eval_label, imp_index

    def fit(
        self,
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
        test_news_file=None,
        test_behaviors_file=None,
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_news_file is not None, evaluate it too.

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_news_file (str): test set.

        Returns:
            obj: An instance of self.
        """
        ite = 0
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            epoch_re_loss = 0
            epoch_infer_loss = 0
            epoch_ctr1_loss = 0
            epoch_ctr2_loss = 0
            
            train_start = time.time()

            tqdm_util = tqdm(
                self.train_iterator.load_data_from_file(
                    train_news_file, train_behaviors_file
                )
            )
            for batch_data_input in tqdm_util:

                step_result = self.train(batch_data_input)
                step_data_loss = step_result[0]
                recom_loss = step_data_loss-step_result[1]-step_result[2]-step_result[3]
                infer_loss = step_result[1]
                ctr1_loss = step_result[2]
                ctr2_loss = step_result[3]
                                
                epoch_loss += step_data_loss
                epoch_re_loss += recom_loss
                epoch_infer_loss += infer_loss
                epoch_ctr1_loss += ctr1_loss
                epoch_ctr2_loss += ctr2_loss
                
                
                step += 1
                ite +=1 
                if step % self.hparams.show_step == 0:
                    tqdm_util.set_description(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}, recommendation_total_loss: {3:.4f} recommendation_loss: {4:.4f} , infer_total_loss: {5:.4f}, infer_loss: {6:.4f},  ctr1_total_loss: {7:.4f}, ctr1_loss: {8:.4f},  ctr2_total_loss: {9:.4f}, ctr2_loss: {10:.4f}".format(
                            step, epoch_loss / step, step_data_loss,epoch_re_loss / step ,recom_loss ,epoch_infer_loss / step, infer_loss, epoch_ctr1_loss / step ,ctr1_loss ,epoch_ctr2_loss / step, ctr2_loss
                        )
                    )                     

            train_end = time.time()
            train_time = train_end - train_start

            eval_start = time.time()

            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step),
                                ("reco_loss", epoch_re_loss / step),
                                ("infer_loss", epoch_infer_loss / step),
                                ("ctr1_loss", epoch_ctr1_loss / step),
                                ("ctr2_loss", epoch_ctr2_loss / step)]
                ]
            )
            eval_info = ''
            if valid_news_file is not None:

                eval_res = self.run_eval(valid_news_file, valid_behaviors_file,'valid')
                eval_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(eval_res.items(), key=lambda x: x[0])
                    ]
                )
            if test_news_file is not None:
                test_res = self.run_eval(test_news_file, test_behaviors_file,'test')
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_news_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )                                            
            )
        return self

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            all_labels: labels after group.
            all_preds: preds after group.

        """

        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_keys, all_labels, all_preds

    def run_eval(self, news_filename, behaviors_file, eval_type='valid'):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        _, group_labels, group_preds = self.run_slow_eval(
            news_filename, behaviors_file, eval_type
        )
        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        return res


    def run_slow_eval(self, news_filename, behaviors_file, eval_type):
        preds = []
        labels = []
        imp_indexes = []
        
        if eval_type=='valid':
            iterator = self.valid_iterator
        else:
            iterator = self.test_iterator

        for batch_data_input in tqdm(
            iterator.load_data_from_file(news_filename, behaviors_file)
        ):
            step_pred, step_labels, step_imp_index = self.eval(batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
            imp_indexes.extend(np.reshape(step_imp_index, -1))

        group_impr_indexes, group_labels, group_preds = self.group_labels(
            labels, preds, imp_indexes
        )
        return group_impr_indexes, group_labels, group_preds