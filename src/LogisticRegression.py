# =============================================================================
# #
# #  Created on Sun Sep 29 10:16:43 2019
# #
# #  CISC684: Group 2
# #
# #  @author: Murugesan Somasundaram <smuruges@udel.edu>
# #           Eric Allen <allenea@udel.edu>
# #           Matthew Walter <mswalter@udel.edu>        
# #           
# #  Command Line Execution:
# #
# #      python main.py MCAP <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <lambda> <iterations>
# #      python main.py MCAP hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/ 0.01 100
# #      python main.py MCAP hw2datasets/dataset2/train/ham/ hw2datasets/dataset2/train/spam/ hw2datasets/dataset2/test/ham/ hw2datasets/dataset2/test/spam/ 0.01 100
# #      python main.py MCAP hw2datasets/dataset3/train/ham/ hw2datasets/dataset3/train/spam/ hw2datasets/dataset3/test/ham/ hw2datasets/dataset3/test/spam/ 0.01 100
# #
# =============================================================================
#IMPORTS
import os,sys
import re
import math
import glob
from collections import Counter
#import pandas as pd
#import numpy as np


def makeDataSet(directory,class_value):
    """Make the dataset, keep the class, and frequency of words together"""
    data = []
    for file in glob.glob(directory+"/*"):
        if os.path.isfile(file):
           with open(file, 'r', encoding='Latin-1') as text_file:
                text = text_file.read()
                # Stores split by words email and email type (0=Ham,1=Spam). count of unique words -  for every email in the dataset
                data.append([text.split(),class_value, freqWords(text)])
    return data


def freqWords(text):
    """Counts frequency for each email"""
    bagsofwords = Counter(re.findall(r'\w+', text))
    return dict(bagsofwords)


def getWords(emails):
    """Get the unique keywords in the 70_training emails"""
    vocabulary = []
    for email in emails:
        for word in email[0]:
            vocabulary.append(word)
    return list(set(vocabulary))


def split70By30(emails):
    """Split dataset where first 70% are training, and 30% validation"""
    split_num = int(len(emails)*0.70)
    lst70 = []; lst30 = []
    count = 0
    for email in emails:
        #LESS THAN 70% MARK
        if count <= split_num:
            lst70.append(email)
            count += 1
        ##OVER 70% MARK
        else:
            lst30.append(email)
            count += 1
    return lst70,lst30


# Learn weights by using gradient ascent
def learnWeights(training, weights_param, iterations, lam, eta):
    """From Soma's code but modified to work"""
    """Learning the weights each iteration these are updated"""
    # Adjust weights num_iterations times
    for x in range(iterations):
        print("Iteration #",x+1)
        # Set all weights in training set vocabulary to be initially 0.0. w0 ('weight_zero') is initially 0.0
        for w in list(weights_param.keys()):
            total = 0.0
            # ...using all training instances
            for i in range(len(training)):
                # y_sample is true y value (classification) of the doc
                y_sample = 0.0
                if training[i][1] == 1:
                    y_sample = 1.0
                # Only add to the sum if the doc contains the token (the count of it would be 0 anyways)
                if w in training[i][2]:
                    total += float(training[i][2][w]) * (y_sample - calculateCondProb(1, weights_param, training[i][2]))
            weights_param[w] += ((eta * total) - (eta * float(lam) * weights_param[w]))


# Calculate conditional probability for the specified doc. Where class_prob is 1|X or 0|X
# 1 is spam and 0 is ham
def calculateCondProb(class_prob, weights_param, email_wordfreq):
    """From Soma's code but modified to work"""
    # Total tokens in doc. Used to normalize word counts to stay within 0 and 1 for avoiding overflow
    lst_keys = list(email_wordfreq.keys())
    if class_prob == 0:
        sum_wx_0 = weights_param['weight_zero']
        for key in lst_keys:   
            if key not in weights_param:
                weights_param[key] = 0.0

            # sum of weights * token count for each token in each document
            sum_wx_0 += weights_param[key] * float(email_wordfreq[key])
        return 1.0 / (1.0 + math.exp(float(sum_wx_0)))

    # Handle 1
    elif class_prob == 1:
        sum_wx_1 = weights_param['weight_zero']
        for key in lst_keys:   
            if key not in weights_param:
                weights_param[key] = 0.0
            # sum of weights * token count for each token in each document
            sum_wx_1 += weights_param[key] * float(email_wordfreq[key])
        return math.exp(float(sum_wx_1)) / (1.0 + math.exp(float(sum_wx_1)))


# Apply algorithm to guess class for specific instance of test set
def applyLogisticRegression(email_wordfreq, weights_param):
    """From Soma's code but modified to work"""
    score = {}
    score[0] = calculateCondProb(0, weights_param, email_wordfreq)
    score[1] = calculateCondProb(1, weights_param, email_wordfreq)
    if score[1] > score[0]:
        return 1
    else:
        return 0
    
    
def MCAP(path_ham_train,path_spam_train,path_ham_test,path_spam_test, lambda_value, iterations):
    """Main Function for MCAP LogReg w/ L2 Reg."""
    for path in [path_ham_train, path_spam_train,path_ham_test, path_spam_test]:
        if os.path.exists(path) == False:
            print("INVALID PATH: PATH NOT FOUND.... Exiting from LogisticRegression.")
            sys.exit(0)
            
    #Learned Parameters
    eta = 0.01
    lambda_value = float(lambda_value)
    iterations = int(iterations)
    
    #HAM (0) AND THEN SPAM (1)
    train_files = [path_ham_train, path_spam_train] 
    test_files = [path_ham_test, path_spam_test]
    C_Bool = [0,1]
    
    #GET DATA AND DIVIDE THE TRAINING DATASETS
    allfirst70 = []; alllast30 = []; data_train = []; data_test = []
    for C in C_Bool:
        train = makeDataSet(train_files[C],C)
        data_train = data_train + train
        test = makeDataSet(test_files[C],C)
        data_test = data_test + test
        emails_70_train,emails_30_validation = split70By30(train)
        allfirst70 = allfirst70 + emails_70_train
        alllast30 = alllast30 + emails_30_validation
    
    #GET ALL WORDS FROM TRAINING DATA
    setWords_train = getWords(allfirst70)
    
    # Set all weights in training set vocabulary to be initially 0.0
    weights = {'weight_zero': 0.0}
    for i in setWords_train:
        weights[i] = 0.0

    # Learn weights -- output to weights
    learnWeights(allfirst70, weights, iterations, lambda_value, eta)
    
    # Apply algorithm on test set
    correct_guesses = 0.0
    for i in range(len(data_test)):
        if applyLogisticRegression(data_test[i][2], weights) == data_test[i][1]:
            correct_guesses += 1.0

    print("Correctly Guessed: " + str(int(correct_guesses)))
    print("Total Guessed: " + str(len(data_test)))
    print("Overall Accuracy: " + str(100.0 * (float(correct_guesses) / len(data_test))))