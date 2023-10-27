import tensorflow as tf
import numpy as np
import pickle

from base_iterator import BaseIterator
from newsrec_utils import word_tokenize, newsample
import string
from data_augmentation import Crop, Mask, Reorder
from random import shuffle
import random



__all__ = ["MINDAllIterator"]


class MINDAllIterator(BaseIterator):

    def __init__(
        self, hparams, npratio=-1, col_spliter="\t", ID_spliter="%",
    ):
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams.batch_size
        self.title_size = hparams.title_size
        self.body_size = hparams.body_size
        self.his_size = hparams.his_size
        self.npratio = npratio
        self.filter_num = hparams.filter_his_num
        self.ctr2_reorder_beta = hparams.ctr2_reorder_beta
        self.ctr2_mask_gamma = hparams.ctr2_mask_gamma
        self.ctr2_crop_tao = hparams.ctr2_crop_tao
        self.aug1 = hparams.aug1
        self.aug2 = hparams.aug2
        
        self.ctr1_loss_weight = hparams.ctr1_loss_weight
        self.ctr1_mask_ratio = hparams.ctr1_mask_ratio


        self.word_dict = self.load_dict(hparams.wordDict_file)
        self.vert_dict = self.load_dict(hparams.vertDict_file)
        self.subvert_dict = self.load_dict(hparams.subvertDict_file)
        self.uid2index = self.load_dict(hparams.userDict_file)
        self.stopwords = self.load_stopwords("stopwords.txt")
        self.punc = string.punctuation
        
        self.base_transform = self._get_base_transformer()
        self.ctr1_base_transform = self._get_ctr1_base_transformer()
                
        print("finish initial Iterator")
               
    
    def load_user_his_pos_pair(self,file):
        embInfo = []
        with open(file, 'rb') as f:
            while True:
                try:
                    l_node = pickle.load(f)
                    embInfo.append(l_node)
                except EOFError:
                     break
        return embInfo
    
    def load_dict(self, file_path):
        """ load pickle file
        Args:
            file path (str): file path
        
        Returns:
            (obj): pickle load obj
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)
        
    def load_stopwords(self, file_path):        
        with open(file_path,"r")as f:
            stopwords = f.readlines()
            f.close()
        stopwords =[w.strip('\n') for w in stopwords]
        return stopwords

    def remove_stopwords(self, title):
        title1=[]
        for t in title:
            if t not in self.stopwords:
                title1.append(t)
        return title1
        
    def init_news(self, news_file):
        """ init news information given news file, such as news_title_index, news_abstract_index.
        Args:
            news_file: path of news file
        """
        self.nid2index = {}
        news_title = [""]
        news_ab = [""]
        news_vert = [""]
        news_subvert = [""]
        
        
        with tf.io.gfile.GFile(news_file, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                    self.col_spliter
                )

                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                title = word_tokenize(title)
                title = self.remove_stopwords(title)
                
                ab = word_tokenize(ab)
                ab = self.remove_stopwords(ab)
                
                news_title.append(title)
                news_ab.append(ab)
                news_vert.append(vert)
                news_subvert.append(subvert)

        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )

        self.news_ab_index = np.zeros((len(news_ab), self.body_size), dtype="int32")
        self.news_vert_index = np.zeros((len(news_vert), 1), dtype="int32")
        self.news_subvert_index = np.zeros((len(news_subvert), 1), dtype="int32")

        for news_index in range(len(news_title)):
            title = news_title[news_index]
            ab = news_ab[news_index]
            vert = news_vert[news_index]
            subvert = news_subvert[news_index]
            for word_index in range(min(self.title_size, len(title))):
                if title[word_index] in self.word_dict:
                    self.news_title_index[news_index, word_index] = self.word_dict[
                        title[word_index].lower()
                    ]
            for word_index_ab in range(min(self.body_size, len(ab))):
                if ab[word_index_ab] in self.word_dict:
                    self.news_ab_index[news_index, word_index_ab] = self.word_dict[
                        ab[word_index_ab].lower()
                    ]
            if vert in self.vert_dict:
                self.news_vert_index[news_index, 0] = self.vert_dict[vert]
            if subvert in self.subvert_dict:
                self.news_subvert_index[news_index, 0] = self.subvert_dict[subvert]

    def init_behaviors(self, behaviors_file):
        """ init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []
        self.user_histories = {}

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                                
                if len(history)<self.filter_num:
                    continue
                
                history = [0] * (self.his_size - len(history)) + history[
                    : self.his_size
                ]
                
                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                 
                if uindex not in self.user_histories:
                    self.user_histories[uindex] = self.news_title_index[history]
                
                impr_index += 1
        print("finish init_behaviors")        
                
                
    def parser_one_line(self, line):
        if self.npratio > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [1] + [0] * self.npratio

                n = newsample(negs, self.npratio)
                candidate_title_index = self.news_title_index[[p] + n]
                candidate_ab_index = self.news_ab_index[[p] + n]
                candidate_vert_index = self.news_vert_index[[p] + n]
                candidate_subvert_index = self.news_subvert_index[[p] + n]
                click_title_index = self.news_title_index[self.histories[line]]
                click_ab_index = self.news_ab_index[self.histories[line]]
                click_vert_index = self.news_vert_index[self.histories[line]]
                click_subvert_index = self.news_subvert_index[self.histories[line]]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])
                
                
                if self.ctr1_loss_weight>0.0:
                    click_title_pos_index = self._ctr1_one_pair_data_augmentation(click_title_index)
                    
                else:
                    click_title_pos_index = np.zeros((self.his_size, 2, self.title_size))
                                        
                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    candidate_ab_index,
                    candidate_vert_index,
                    candidate_subvert_index,
                    click_title_index,
                    click_ab_index,
                    click_vert_index,
                    click_subvert_index,                    
                    click_title_pos_index,
                )

        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [label]

                candidate_title_index.append(self.news_title_index[news])
                click_title_index = self.news_title_index[self.histories[line]]

                candidate_title_index = self.news_title_index[news]
                candidate_ab_index = self.news_ab_index[news]
                candidate_vert_index = self.news_vert_index[news]
                candidate_subvert_index = self.news_subvert_index[news]
                click_title_index = self.news_title_index[self.histories[line]]
                click_ab_index = self.news_ab_index[self.histories[line]]
                click_vert_index = self.news_vert_index[self.histories[line]]
                click_subvert_index = self.news_subvert_index[self.histories[line]]
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])
                click_title_pos_index = []
                

                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_title_index,
                    candidate_ab_index,
                    candidate_vert_index,
                    candidate_subvert_index,
                    click_title_index,
                    click_ab_index,
                    click_vert_index,
                    click_subvert_index,
                    click_title_pos_index,
                )

    def load_data_from_file(self, news_file, behavior_file):

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_title_indexes = []
        candidate_ab_indexes = []
        candidate_vert_indexes = []
        candidate_subvert_indexes = []
        click_title_indexes = []
        click_ab_indexes = []
        click_vert_indexes = []
        click_subvert_indexes = []
        click_title_pos_indexes = []

        cnt = 0

        indexes = np.arange(len(self.labels))

        if self.npratio > 0:
            np.random.shuffle(indexes)

        for index in indexes:
            for (
                label,
                impr_index,
                user_index,
                candidate_title_index,
                candidate_ab_index,
                candidate_vert_index,
                candidate_subvert_index,
                click_title_index,
                click_ab_index,
                click_vert_index,
                click_subvert_index,                
                click_title_pos_index,
            ) in self.parser_one_line(index):
                candidate_title_indexes.append(candidate_title_index)
                candidate_ab_indexes.append(candidate_ab_index)
                candidate_vert_indexes.append(candidate_vert_index)
                candidate_subvert_indexes.append(candidate_subvert_index)
                click_title_indexes.append(click_title_index)
                click_ab_indexes.append(click_ab_index)
                click_vert_indexes.append(click_vert_index)
                click_subvert_indexes.append(click_subvert_index)
                imp_indexes.append(impr_index)
                user_indexes.append(user_index)
                label_list.append(label)
                click_title_pos_indexes.append(click_title_pos_index)

                cnt += 1                
                if cnt >= self.batch_size:
                    augmented_clicked_title_indexes,users_augmented_mask = self.get_augment_data(user_indexes)
                    
                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        user_indexes,
                        candidate_title_indexes,
                        candidate_ab_indexes,
                        candidate_vert_indexes,
                        candidate_subvert_indexes,
                        click_title_indexes,
                        click_ab_indexes,
                        click_vert_indexes,
                        click_subvert_indexes,
                        click_title_pos_indexes,
                        augmented_clicked_title_indexes,
                        users_augmented_mask,
                    )
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    candidate_title_indexes = []
                    candidate_ab_indexes = []
                    candidate_vert_indexes = []
                    candidate_subvert_indexes = []
                    click_title_indexes = []
                    click_ab_indexes = []
                    click_vert_indexes = []
                    click_subvert_indexes = []
                    click_title_pos_indexes = []
                    cnt = 0
                    

    def _convert_data(
        self,
        label_list,
        imp_indexes,
        user_indexes,
        candidate_title_indexes,
        candidate_ab_indexes,
        candidate_vert_indexes,
        candidate_subvert_indexes,
        click_title_indexes,
        click_ab_indexes,
        click_vert_indexes,
        click_subvert_indexes,
        click_title_pos_indexes,
        augmented_clicked_title_indexes,
        users_augmented_mask,
    ):

        labels = np.asarray(label_list, dtype=np.float32)
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(
            candidate_title_indexes, dtype=np.int64
        )
        candidate_ab_index_batch = np.asarray(candidate_ab_indexes, dtype=np.int64)
        candidate_vert_index_batch = np.asarray(candidate_vert_indexes, dtype=np.int64)
        candidate_subvert_index_batch = np.asarray(
            candidate_subvert_indexes, dtype=np.int64
        )
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        click_ab_index_batch = np.asarray(click_ab_indexes, dtype=np.int64)
        click_vert_index_batch = np.asarray(click_vert_indexes, dtype=np.int64)
        click_subvert_index_batch = np.asarray(click_subvert_indexes, dtype=np.int64)
        click_title_pos_index_batch = np.asarray(click_title_pos_indexes, dtype=np.int64)
        
        
        augmented_clicked_title_indexes_batch = np.asarray(
            augmented_clicked_title_indexes, dtype=np.int64
        )
        users_augmented_mask_batch = np.asarray(users_augmented_mask, dtype=np.float64
        )
                
        
        return {
            "impression_index_batch": imp_indexes,
            "user_index_batch": user_indexes,
            "clicked_title_batch": click_title_index_batch,
            "clicked_ab_batch": click_ab_index_batch,
            "clicked_vert_batch": click_vert_index_batch,
            "clicked_subvert_batch": click_subvert_index_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_ab_batch": candidate_ab_index_batch,
            "candidate_vert_batch": candidate_vert_index_batch,
            "candidate_subvert_batch": candidate_subvert_index_batch,
            "labels": labels,
            "clicked_title_pos_batch": click_title_pos_index_batch,
            "augmented_clicked_title_batch":augmented_clicked_title_indexes_batch,
            "users_augmented_mask_batch":users_augmented_mask_batch,            
        }


    def load_impression_from_file(self, behaivors_file):

        if not hasattr(self, "histories"):
            self.init_behaviors(behaivors_file)

        indexes = np.arange(len(self.labels))

        for index in indexes:
            impr_label = np.array(self.labels[index], dtype="int32")
            impr_news = np.array(self.imprs[index], dtype="int32")

            yield (
                self.impr_indexes[index],
                impr_news,
                self.uindexes[index],
                impr_label,
            )                        
            
    def _ctr1_one_pair_data_augmentation(self, clicked_title):
               
        augmented_seqs = []
        for input_ids in clicked_title:
            augmented_title = []
            for i in range(2):
                input_ids_1 = input_ids[input_ids!=0]
                if input_ids_1.shape[0]==0:
                    augmented_input = np.array([])
                else:
                    ratio = int(np.ceil(self.ctr1_mask_ratio*input_ids_1.shape[0]))
                    mask_words = random.sample(list(input_ids_1), k = ratio)
                    mask_matrix = []
                    for word in input_ids_1:
                        if word in mask_words:
                            mask_matrix.append(0)
                        else:
                            mask_matrix.append(1)
                    mask_matrix = np.asarray(mask_matrix, dtype="int32")
                    augmented_input = np.multiply(input_ids_1, mask_matrix)
                           
                pad_len = self.title_size - augmented_input.shape[0]
                augmented_input = np.concatenate((augmented_input,[0] * pad_len), axis=0)
                augmented_input = augmented_input[-self.title_size:]

                assert augmented_input.shape[0] == self.title_size            
                cur_tensors = augmented_input
                augmented_title.append(cur_tensors)
            augmented_title = np.asarray(augmented_title,dtype=np.int64)  
            augmented_seqs.append(augmented_title)
        augmented_seqs = np.asarray(augmented_seqs,dtype=np.int64)
                       
        return augmented_seqs
    
    
    def _one_pair_data_augmentation(self, input_ids):
        input_ids = np.asarray(input_ids,dtype='int32')
        augmented_seqs = []
        for i in range(2):
            
            augmented_input = self.base_transform[i](input_ids)       
            pad_len = self.his_size - augmented_input.shape[0]
            zero_vec = np.zeros(input_ids.shape, dtype="int32")     
            augmented_input = np.concatenate((zero_vec * pad_len, augmented_input), axis=0)

            augmented_input = augmented_input[-self.his_size:]

            assert augmented_input.shape[0] == self.his_size
            cur_tensors = augmented_input
            augmented_seqs.append(cur_tensors)
       
        return augmented_seqs

    def get_augment_data(self, users):
        diff_user = np.unique(users)        
        user_cnt = len(diff_user)
        users_augmented_mask = [1]*user_cnt+[0]*(self.batch_size-user_cnt)
        
        users_augmented_sequence = []
        randuser = random.sample(self.user_histories.keys()-diff_user, self.batch_size-user_cnt)
        user_indexs = np.concatenate((diff_user, randuser), axis=0) 
               
        users_augmented_sequence = [self._one_pair_data_augmentation(self.user_histories[uid]) for uid in user_indexs] 
                    
        return users_augmented_sequence,users_augmented_mask
    
    
    def _get_base_transformer(self):
                
        aug1 = self.aug1
        aug2 = self.aug2
        transform1 = None
        transform2 = None
        if aug1 == "record":
            transform1 = Reorder(beta=self.ctr2_reorder_beta)
        elif aug1 == 'mask':
            transform1 = Mask(gamma=self.ctr2_mask_gamma)
        elif aug1 == 'crop':
            transform1 = Crop(tao=self.ctr2_crop_tao)

        if aug2 == "record":
            transform2 = Reorder(beta=self.ctr2_reorder_beta)
        elif aug2 == 'mask':
            transform2 = Mask(gamma=self.ctr2_mask_gamma)
        elif aug2 == 'crop':
            transform2 = Crop(tao=self.crop_tao)
        
        base_transform = [transform1, transform2]
        
        return base_transform

    def _get_ctr1_base_transformer(self):
        transform1 = None
        transform2 = None
        transform1 = Mask(gamma=self.ctr1_mask_ratio)
        transform2 = Mask(gamma=self.ctr1_mask_ratio)
        
        base_transform = [transform1, transform2]
        
        return base_transform

    
    def load_news_from_file(self, news_file):
        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        news_indexes = []
        candidate_title_indexes = []
        candidate_ab_indexes = []
        candidate_vert_indexes = []
        candidate_subvert_indexes = []

        for index in range(len(self.news_title_index)):
            news_indexes.append(index)
            candidate_title_indexes.append(self.news_title_index[index])
            candidate_ab_indexes.append(self.news_ab_index[index])
            candidate_vert_indexes.append(self.news_vert_index[index])
            candidate_subvert_indexes.append(self.news_subvert_index[index])
        
        return np.asarray(
            candidate_title_indexes, dtype=np.int64
        )
