import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
import numpy as np
import yaml
import pickle as pkl
import random
import re


def check_type(config):

    int_parameters = [
        "word_size",
        "his_size",
        "title_size",
        "body_size",
        "npratio",
        "word_emb_dim",
        "attention_hidden_dim",
        "epochs",
        "batch_size",
        "show_step",
        "save_epoch",
        "head_num",
        "head_dim",
        "user_num",
        "filter_num",
        "window_size",
        "user_emb_dim",
        "vert_emb_dim",
        "subvert_emb_dim",
        "channels",
        "filter_his_num",
    ]
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("Parameters {0} must be int".format(param))

    float_parameters = [
        "learning_rate",                  
        "dropout",
        "purpose_router_tp",
        "infer_loss_weight",
        "ctr2_crop_tao",
        "ctr2_mask_gamma",
        "ctr2_reorder_beta",
        "ssa_r",
        "ctr1_mask_ratio",
    ]
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("Parameters {0} must be float".format(param))

    str_parameters = [
        "wordEmb_file",
        "wordDict_file",
        "userDict_file",
        "vertDict_file",
        "subvertDict_file",
        "method",
        "loss",
        "optimizer",
        "cnn_activation",
        "dense_activation" ,
        "aug1",
        "aug2",
    ]
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("Parameters {0} must be str".format(param))

    list_parameters = ["layer_sizes", "activation"]
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("Parameters {0} must be list".format(param))

    bool_parameters = ["support_quick_scoring"]
    for param in bool_parameters:
        if param in config and not isinstance(config[param], bool):
            raise TypeError("Parameters {0} must be bool".format(param))


def check_nn_config(f_config):
    required_parameters = [
        "title_size",
        "his_size",
        "wordEmb_file",
        "wordDict_file",
        "userDict_file",
        "npratio",
        "data_format",
        "word_emb_dim",
        "head_num",
        "head_dim",
        "attention_hidden_dim",
        "loss",
        "data_format",
        "dropout",
        "channels",
        "purpose_router_tp",
        "infer_loss_weight",
        "ctr1_loss_weight",
        "ctr2_loss_weight",
        "filter_his_num",
        "aug1",
        "aug2",
        "ctr2_crop_tao",
        "ctr2_mask_gamma",
        "ctr2_reorder_beta",
        "ssa_r",
        "ctr1_mask_ratio",
    ]

    
    # check required parameters
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("Parameters {0} must be set".format(param))

    
    if f_config["data_format"] != "news":
        raise ValueError(
            "For nrms and naml model, data format must be 'news', but your set is {0}".format(
                f_config["data_format"]
            )
        )


    check_type(f_config)


def create_hparams(flags):
    return tf.contrib.training.HParams(
        # data
        data_format=flags.get("data_format", None),
        iterator_type=flags.get("iterator_type", None),
        support_quick_scoring=flags.get("support_quick_scoring", False),
        wordEmb_file=flags.get("wordEmb_file", None),
        wordDict_file=flags.get("wordDict_file", None),
        userDict_file=flags.get("userDict_file", None),
        vertDict_file=flags.get("vertDict_file", None),
        subvertDict_file=flags.get("subvertDict_file", None),
        # models
        title_size=flags.get("title_size", None),
        body_size=flags.get("body_size", None),
        word_emb_dim=flags.get("word_emb_dim", None),
        word_size=flags.get("word_size", None),
        user_num=flags.get("user_num", None),
        vert_num=flags.get("vert_num", None),
        subvert_num=flags.get("subvert_num", None),
        his_size=flags.get("his_size", None),
        npratio=flags.get("npratio"),
        dropout=flags.get("dropout", 0.0),
        attention_hidden_dim=flags.get("attention_hidden_dim", 200),
        filter_his_num = flags.get("filter_his_num",5),
        head_num=flags.get("head_num", 4),
        head_dim=flags.get("head_dim", 100),
        cnn_activation=flags.get("cnn_activation", None),
        dense_activation=flags.get("dense_activation", None),
        filter_num=flags.get("filter_num", 200),
        window_size=flags.get("window_size", 3),
        vert_emb_dim=flags.get("vert_emb_dim", 100),
        subvert_emb_dim=flags.get("subvert_emb_dim", 100),
        user_emb_dim=flags.get("user_emb_dim", 50),
        channels = flags.get("channels", 3),
        purpose_router_tp = flags.get("purpose_router_tp",0.1),
        infer_loss_weight = flags.get("infer_loss_weight",1.0),
        ctr1_loss_weight = flags.get("ctr1_loss_weight",1.0),
        ctr2_loss_weight = flags.get("ctr2_loss_weight",1.0),
        aug1 = flags.get("aug1","mask"),
        aug2 = flags.get("aug2","mask"),
        ctr2_crop_tao = flags.get("ctr2_crop_tao",0.2),
        ctr2_mask_gamma = flags.get("ctr2_mask_gamma",0.5),
        ctr2_reorder_beta = flags.get("ctr2_reorder_beta", 0.2),
        ssa_r = flags.get("ssa_r",0.5), 
        ctr1_mask_ratio = flags.get("ctr1_mask_ratio",0.1),
        # train
        learning_rate=flags.get("learning_rate", 0.001),
        loss=flags.get("loss", None),
        optimizer=flags.get("optimizer", "adam"),
        epochs=flags.get("epochs", 10),
        batch_size=flags.get("batch_size", 1),
        # show info
        show_step=flags.get("show_step", 1),
        metrics=flags.get("metrics", None),
    )


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare the model hyperparameters and check that all have the correct value.

    Args:
        yaml_file (str): YAML file as configuration.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    config.update(kwargs)

    check_nn_config(config)
    return create_hparams(config)


def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def newsample(news, ratio):
    """ Sample ratio samples from news list. 
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number
    
    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)

def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.
    
    Returns:
        np.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (np.ndarray): ground-truth labels.
        y_score (np.ndarray): predicted labels.

    Returns:
        np.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    
    FIXME: 
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res
    

    
def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.
    
    Args:
        config (dict): Configuration loaded from a yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config

def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception as e:  # for other exceptions
        raise IOError("load {0} error!".format(filename))

def load_dict(filename):
    """Load the vocabularies.

    Args:
        filename (str): Filename of user, item or category vocabulary.

    Returns:
        dict: A saved vocabulary.
    """
    with open(filename, "rb") as f:
        f_pkl = pkl.load(f)
        return f_pkl
        
