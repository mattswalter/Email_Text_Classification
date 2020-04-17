# =============================================================================
# #
# #  Created on Tue Oct  1 10:29:08 2019
# #
# #  CISC684: Group 2
# #
# #  @author: Matthew Walter <mswalter@udel.edu>
# #           Eric Allen <allenea@udel.edu>
# #           Murugesan Somasundaram <smuruges@udel.edu>
# #           
# #  Command Line Execution:
# #      python main.py BAYES <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test>
# #      python main.py BAYES hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/
# #      python main.py BAYES hw2datasets/dataset2/train/ham/ hw2datasets/dataset2/train/spam/ hw2datasets/dataset2/test/ham/ hw2datasets/dataset2/test/spam/
# #      python main.py BAYES hw2datasets/dataset3/train/ham/ hw2datasets/dataset3/train/spam/ hw2datasets/dataset3/test/ham/ hw2datasets/dataset3/test/spam/
# =============================================================================
#IMPORTS
import os, sys
import glob
import pandas as pd
import numpy as np


def word_freq(loc):
    """Calculate frequency of words in a class """
    data = []
    for file in glob.glob(loc+"/*"):
        with open(file,'r', encoding="Latin-1") as myfile:
            data.append(myfile.read())
            
    #Store emails in the dataframe
    df = pd.DataFrame(data,index=None)
    #Number of emails (shape)
    shape = df.shape[0]
    # Get frequency of each unique word    
    freq = pd.Series(np.concatenate([x.split() for x in df[0]])).value_counts()
    freq.to_frame()
    return freq, shape


def test_emails(loc):
    """Open test data"""
    data = []
    for file in glob.glob(loc+"/*"):
        with open(file,'r', encoding="Latin-1") as myfile:
            data.append(myfile.read())
            
    # Store words in dataframe
    df = pd.DataFrame(data,columns=["emails"],index=None)
    return df


def count_words(ham, spam):
    """Calculate probability that any email is in a class (total emails in class / total emails)"""
    # Total words
    s_words = spam.sum()
    h_words = ham.sum()
    # Count of unique words in data set by adding unique words in ham + unique words in spam
    c_dict = {}    
    c_dict.update(ham)
    c_dict.update(spam)
    u_words = len(c_dict)
    return s_words, h_words, u_words


def prob_class(spam, ham, l):  
    """Calculates probability of a spam or ham email occuring: one class count / total class count"""
    total = spam + ham
    t_spam = spam/total
    t_ham = ham/total
    if l == 1:
        return t_ham
    if l == 0:
        return t_spam


def bayes_equation(words, spam, ham, p):
    """Calculates Bayes """
    #Matt: Not sure what this is -- Eric
    #Second half of bayes equation: total words in that class + distinct words in dataset
    #prob_spam = words[0]+words[2]
    #prob_ham = words[1]+words[2]

    # First half of bayes equation: frequency of word in class + 1
    bayes_spam = spam+1
    bayes_ham = ham+1
    # Save into 2 dictionaries for probability of spam and ham
    p_spam = bayes_spam.to_dict()
    p_ham = bayes_ham.to_dict()
    if p == 0:
        return p_spam 
    if p == 1:
        return p_ham
    

def bayes_test(freq_ham, freq_spam, emails, idx, c, words, spam_ham_data):
    """classify test data"""
    # Count frequency of words in each email
    c_spam = words[0]+words[2]
    c_ham = words[1]+words[2]
    #Each email
    s = emails.emails[idx:idx+1]
    #Count of words for each email
    s1 = pd.Series(np.concatenate([x.split() for x in s])).value_counts()
    #Dictionary of the frequency of words. Remember value per key
    d = s1.to_dict()
    calc_spam = 0
    calc_ham = 0
    calc_spam1 = 0
    calc_ham1 = 0
    for key in d:
        ## HAM
        if key in spam_ham_data['prob_ham']:
            prob1 = spam_ham_data['prob_ham'][key]
            #power = d[key]
            calc_ham1 = np.log(prob1/c_ham)
            calc_ham = calc_ham + calc_ham1
            
        else: #keyh not in prob_ham
            prob1 = 1
            #power = d[key]
            calc_ham1 = np.log(prob1/c_ham)
            calc_ham = calc_ham + calc_ham1
        
        ## SPAM
        if key in spam_ham_data['prob_spam']:
            prob2 = spam_ham_data['prob_spam'][key]
            #power = d[key]
            calc_spam1 = np.log(prob2/c_spam)
            calc_spam = calc_spam + calc_spam1
            
        else: # keyh not in prob_spam:
            prob2 = 1
            #power = d[key]
            calc_spam1 = np.log(prob2/c_spam)
            calc_spam = calc_spam + calc_spam1
         
    spam = calc_spam + spam_ham_data['spam_p_class']
    ham = calc_ham + spam_ham_data['ham_p_class']
    
    if ham >= spam:
        if c ==1:
            spam_ham_data['ham_r'] = 1 + spam_ham_data['ham_r']
        if c == 0:
            spam_ham_data['spam_w'] = 1 + spam_ham_data['spam_w']
            
    elif spam > ham:
        if c == 1:
            spam_ham_data['ham_w'] = 1 + spam_ham_data['ham_w']
        if c == 0:
            spam_ham_data['spam_r'] = 1 + spam_ham_data['spam_r']
    
    ##TODO: PROBLEM WITH THE PAPER? What if they are equal spam and ham.... Doesn't happen here
    # But I would presume that if they are equal then it is ham.... 

    if c == 1:
        return spam_ham_data['ham_r'], spam_ham_data['ham_w']
    elif c == 0:
        return spam_ham_data['spam_r'], spam_ham_data['spam_w']
    else: # NOT 0 or 1!!!
        return d


        
def bayes(path_ham_train, path_spam_train,path_ham_test, path_spam_test):
    """Main function"""
    #dataset = "dataset1"#sys.argv[1]
    #dirpath = os.getcwd()
    #TRAIN
    #path_ham_train = os.path.join(dirpath, "hw2datasets",dataset,"train","ham")#'\\' + dataset + '\\train\\ham'
    #path_spam_train = os.path.join(dirpath,"hw2datasets", dataset,"train","spam")#'\\' + dataset + '\\train\spam'
    #TEST
    #path_ham_test= os.path.join(dirpath, "hw2datasets", dataset,"test","ham")#'\\' + dataset + '\\test\\ham'
    #path_spam_test= os.path.join(dirpath, "hw2datasets", dataset,"test","spam")#'\\' + dataset + '\\test\\spam'
    for path in [path_ham_train, path_spam_train,path_ham_test, path_spam_test]:
        if os.path.exists(path) == False:
            print("INVALID PATH: PATH NOT FOUND.... Exiting from bayes.")
            sys.exit(0)
            
    #PART 1 - Train
    #Get word frequency
    ham_Freq, num_ham_emails = word_freq(path_ham_train)
    spam_Freq, num_spam_emails = word_freq(path_spam_train)
    
    #Count Words
    words = count_words(ham_Freq, spam_Freq)
    
    #Get Probability
    ham_p_class = prob_class(num_spam_emails, num_ham_emails, 1)
    spam_p_class = prob_class(num_spam_emails, num_ham_emails, 0)
    
    #Bayes
    prob_spam = bayes_equation(words, spam_Freq, ham_Freq, 0)
    prob_ham = bayes_equation(words, spam_Freq, ham_Freq, 1)
    
    #PART 2 - Test
    ham_test = test_emails(path_ham_test)
    spam_test = test_emails(path_spam_test)
    
    spam_ham_data = {'prob_spam':prob_spam, 'prob_ham':prob_ham,\
                     'spam_p_class':spam_p_class,'ham_p_class':ham_p_class,\
                     'spam_r':0,'ham_r':0, 'spam_w':0, 'ham_w':0}
    
    for index in range(len(spam_test)):    
        spam_emails = bayes_test(ham_Freq, spam_Freq, spam_test, index, 0, words, spam_ham_data)
        
    spam_right = spam_emails[0] # True
    spam_wrong = spam_emails[1] # False
    
    for index in range(len(ham_test)):  
        ham_emails = bayes_test(ham_Freq, spam_Freq, ham_test, index, 1, words, spam_ham_data)
        
    ham_right = ham_emails[0] # True
    ham_wrong = ham_emails[1] # False
    
    #Calculate Accuracy
    right = ham_right + spam_right
    wrong = ham_wrong + spam_wrong
    accuracy = (right/(wrong+right))*100
    
    print("Correctly Guessed: " + str(right))
    print("Inorrectly Guessed: " + str(wrong))
    print("Overall Accuracy: " + str(accuracy))