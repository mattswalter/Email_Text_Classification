# =============================================================================
# #
# #  Created on Sun Oct 6 10:16:43 2019
# #
# #  CISC684: Group 2
# #
# #  @author: Eric Allen <allenea@udel.edu>
# #           Matthew Walter <mswalter@udel.edu>        
# #           Murugesan Somasundaram <smuruges@udel.edu>
# #
# #  Command Line Execution:
# #
# #
# =============================================================================
#IMPORTS
import os,sys
import re
#import math
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


# =============================================================================
# ================= NEW FOR PERCEPTRON=========================================
# =============================================================================
# =============================================================================


def learnWeights(training_set, weights, iterations, eta):
    """learns weights using the perceptron training rule"""
    # Adjust weights num_iterations times
    for i in range(iterations):
        #print("Iteration #",i+1)
        # Go through all training emails and update weights
        for d in range(len(training_set)):
            weight_sum = weights['weight_zero']
            for f in training_set[d][2]:
                if f not in weights:
                    weights[f] = 0.0
                weight_sum += weights[f] * training_set[d][2][f]
            perceptron_output = 0.0
            if weight_sum > 0:
                perceptron_output = 1.0
            target_value = 0.0
            if training_set[d][1] == 1:
                target_value = 1.0
            # Update all weights
            for w in training_set[d][2]:
                weights[w] += float(eta) * float((target_value - perceptron_output)) * \
                              float(training_set[d][2][w])


def apply(weights, data_test):
    """test accuracy of test dataset"""
    weight_sum = weights['weight_zero']
    lst_keys = list(data_test.keys())
    for key in lst_keys:
        if key not in weights:
            weights[key] = 0.0
        weight_sum += weights[key] * data_test[key]
    if weight_sum > 0:
        return 1 #SPAM
    else:
        return 0 #HAM
    
    
def PERCEPTRON(path_ham_train,path_spam_train,path_ham_test,path_spam_test, iterations):
    """MAIN FUNCTION FOR PERCEPTRON"""
    for path in [path_ham_train, path_spam_train,path_ham_test, path_spam_test]:
        if os.path.exists(path) == False:
            print("INVALID PATH: PATH NOT FOUND.... Exiting from perceptron.")
            sys.exit(0)
            
    #Learned Parameters
    eta = 0.01
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

    #Store weights in a dictionary.... Start at 1.0
    weights = {'weight_zero': 1.0}

    for i in setWords_train:
        weights[i] = 0.0

    # Learn weights using the training_set
    learnWeights(allfirst70, weights, iterations, eta)
   
    #Test against the test dataset
    correct_guesses = 0
    for i in range(len(data_test)):
        guess = apply(weights, data_test[i][2])
        if guess == 1:
            if data_test[i][1] == 1:
                correct_guesses += 1
        if guess == 0:
            if data_test[i][1] == 0:
                correct_guesses += 1

    print("Correctly Guessed: " + str(int(correct_guesses)))
    print("Total Guessed: " + str(len(data_test)))
    print("Overall Accuracy: " + str(100.0 * (float(correct_guesses) / len(data_test))))