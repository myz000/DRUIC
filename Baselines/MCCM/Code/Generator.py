from tensorflow.keras.utils import Sequence
import numpy as np 

class get_hir_train_generator(Sequence):
    def __init__(self,news_scoring,clicked_news,user_id, news_id, label, batch_size,news_freq):
        self.news_emb = news_scoring
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label
        
        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]
        self.news_freq = news_freq
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __get_news(self,docids):
        news_emb = self.news_emb[docids]

        return news_emb
    
    def random_user(self,clicked_ids):
        clicked_p = self.news_freq[clicked_ids]
        aug_user = []
        ratio = int(0.25*clicked_ids.shape[1])
        for i in range(int(clicked_ids.shape[0])):
            clicked_p[i]/=np.sum(clicked_p[i])
            index = [np.random.choice(clicked_ids[i], p = clicked_p[i].ravel()) for aa in range(ratio)]
            mask_matrix = []
            for word in clicked_ids[i]:
                if word in index:
                    mask_matrix.append(0)
                else:
                    mask_matrix.append(1)
            mask_matrix = np.asarray(mask_matrix, dtype="int32")
            augmented_input = np.multiply(clicked_ids[i], mask_matrix)
            aug_user.append(augmented_input)
        aug_user = np.asarray(aug_user, dtype="int32")
        return aug_user
        
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        label = self.label[start:ed]
        
        doc_ids = self.doc_id[start:ed]
        title= self.__get_news(doc_ids)
        
        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        user_title = self.__get_news(clicked_ids)
        
        aug_clicked_ids = self.random_user(clicked_ids)
        aug_user_title = self.__get_news(aug_clicked_ids)
        
        return ([title, user_title, aug_user_title],[label])
    
class get_user_generator(Sequence):
    def __init__(self, news_scoring, userids, clicked_news,batch_size):
        self.userids = userids
        self.news_scoring = news_scoring
        self.clicked_news = clicked_news

        self.batch_size = batch_size
        self.ImpNum = self.clicked_news.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self,docids):
        news_scoring = self.news_scoring[docids]
        
        return news_scoring
              
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        
        userisd = self.userids[start:ed]
        clicked_ids = self.clicked_news[userisd]

        user_title = self.__get_news(clicked_ids)

        return user_title