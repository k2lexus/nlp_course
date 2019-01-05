# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:47:45 2019

@author: LEX
"""

import time
import math
import torch
import os
import torch.onnx
from model import Languagemodel
from utils.data_utils import Vocab, Txtfile, Data2tensor, SaveloadHP, seqPAD, PAD, EOS, SOS
from utils.core_nns import RNNModel


# Load trained model
def load_model(model_source, use_cuda=False):
    """ Load pretrained model from source
            - model_source: link to '.args' file
            - use_cuda: set it to True if you have GPU
        Return: model, vocab
    """
    #model_args_source = './results/lm.args'
    model_args = SaveloadHP.load(model_source)
    model_args.use_cuda = use_cuda
    language_model = Languagemodel(model_args)
    language_model.model.load_state_dict(torch.load(model_args.trained_model))

    return language_model.model, model_args.vocab


def rev_gen( model, vocab, start_word=SOS):
    """ Generate a review starts with 'start_word', ends with '</s>'
    """
    print('Generating sample review .....................')
    with torch.no_grad():
        word_idx = vocab.w2i[start_word]
        all_words = []
        all_words.append(start_word)
        while word_idx != vocab.w2i[EOS]:
            word_tensor = Data2tensor.idx2tensor([[word_idx]])
            hidden = model.init_hidden(word_tensor.size(0))
            output, hidden = model(word_tensor, hidden)
            label_prob, label_pred = model.inference(output)
            word_idx = label_pred.data[0][0].data.numpy()[0]
            all_words.append(vocab.i2w[word_idx])


        return ' '.join(all_words)
            
def wd_pred(model, vocab, sentence):
    """ Predict next word
    """
    with torch.no_grad():
        words = sentence.split(' ')
        for i, word in enumerate(words):
            # transform word to tensor
            word_idx = vocab.w2i[word]
            word_tensor = Data2tensor.idx2tensor([[word_idx]])
            if i == 0:
                hidden = model.init_hidden(word_tensor.size(0))
            output, hidden = model(word_tensor, hidden)
        
    label_prob, label_pred = model.inference(output)
    word_idx = label_pred.data[0][0].data.numpy()[0]

    return vocab.i2w[word_idx]
                
            
        
    
        
        
        
        
        
        
        
        
        
        
        
        













        
        
        