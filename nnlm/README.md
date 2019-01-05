# Neural Network Language Model
## Introduction
- A general package to learn neural network language models. They are:
  + RNN (BiRNN), deep-RNN (deep-biRNN)
  + GRU (BiGRU), deep-GRU (deep-biGRU)
  + LSTM (BiLSTM), deep-LSTM (deep-biLSTM)

## Review-domain Language Model
We will train a neural network language model using reviews on Yelp dataset. Where:  
- Inputs in the *dataset* folder:
  + Train_data (*./dataset/train.small.txt*): used for training a model
  + Dev_data (*./dataset/val.small.txt*): used for validating a model
  + Test_data (*./dataset/test.small.txt*): used for testing a model
- Outputs in the *results* folder:
  + A trained-model file *lm.m*
  + A model-argument file *lm.args*
- Applications:
  + Text generation: generate a restaurant review 
    - P(qj)=P(w1,w2,...,wn)
  + Text recommendation: recommend next words given a context
    - P(wi|wi-1,...,w1)

## Folder Structure
  1. model.py: main script to train the model
  2. predict.py: an interface to generate a review and recommend words given previous words
  3. generate.py (optional): a function to generate a list of reviews avd save into a file
  4. utils/
    - core_nns.py: a script to design the neural network architecture
    - data_utils.py: a script to load data and convert to indices
  5. results/: a folder to save the train model
    - *.m: a trained-model file
    - *.args: a model-argument file
## Package workflow
1. Trained mode
- Design a model architecture (mainly done in utils/core_nns.py)
- Load training and testing data map to indices (mainly done in utils/data_utils.py)
- Feed data to model to train it (mainly done in model.py)
- Save model into files (mainly done in model.py)
- Stop the training process (mainly done in model.py) when  
    + having no improvement on testing data or;
    + exceeding the maximum number of epochs
2. Predicted mode
- Load the save files at training phase
- Write ``predict.py`` file to predict the next words given some initial words: P(wi|wi-1,...,w1) and generate a piece of text.

## Project usage
1. Download this repository: git clone https://github.com/duytinvo/DL_NLP.git
2. Train the model:
  - Change the current directory to "week3/nnlm"
  - Run this command:
    ```
    python model.py [--use_cuda]
    ```
  - See other parameters, this command:
    ```
    python model.py -h
    ```

## Assignment

1. Theory questions:  
    a. What is the purpose of the function *bptt_batch()*  and *repackage_hidden()* in *model.py*  
    b. Describe an overview procedure of the function *train()* in *model.py*  
    
2. Don't train the model. In this assignment, we focus on writing an inference of a neural network using available 
pre-trained model. Let's write an inference file *predict.py* containing three functions:  
    a. **load_model()**: Load saved argument file and model file  
    b. **rev_gen()**: Generate a review starting from *SOS* until reaching *EOS*  
    c. **wd_pred()**: Predict a word given some previous words 
