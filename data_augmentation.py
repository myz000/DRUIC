import random
import numpy as np

__all__ = ["Crop", "Mask", "Reorder"]

class Crop(object):
    """Randomly crop a subseq from the original sequence"""
    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        # sequence : Tensor
        copied_sequence = sequence
        seq_length = int(sequence.shape[0])
        sub_seq_length = int(self.tao*seq_length)
        #randint generate int x in range: a <= x <= b
        start_index = random.randint(0, seq_length-sub_seq_length-1)
        if sub_seq_length<1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index:start_index+sub_seq_length]
            return cropped_seq

class Mask(object):
    """Randomly mask k items given a sequence"""
    def __init__(self, gamma=0.7):
        self.gamma = gamma

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        # sequence : Tensor
        copied_sequence = sequence 
        seq_length = int(copied_sequence.shape[0])
        mask_nums = int(self.gamma*seq_length)
        zero_vec = np.zeros_like(sequence[0])
        one_vec = np.ones_like(sequence[0])

        mask_idx = random.sample([i for i in range(seq_length)], k = mask_nums)
        mask_matrix = []
        for i in range(seq_length):
            if i in mask_idx:
                mask_matrix.append(zero_vec)
            else:
                mask_matrix.append(one_vec)
        
        mask_matrix = np.asarray(mask_matrix, dtype="int32")
        copied_sequence = np.multiply(copied_sequence, mask_matrix)
        return copied_sequence

class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""
    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        # sequence : Tensor
        copied_sequence = sequence
        seq_len = int(copied_sequence.shape[0])
        sub_seq_length = int(self.beta*seq_len)
        start_index = random.randint(0, seq_len-sub_seq_length-1)
        sub_seq = copied_sequence[start_index:start_index+sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = np.concatenate((copied_sequence[:start_index], sub_seq), axis=0)
        reordered_seq = np.concatenate((reordered_seq, copied_sequence[start_index+sub_seq_length:]), axis=0)
        assert copied_sequence.shape[0] == reordered_seq.shape[0]
        return reordered_seq