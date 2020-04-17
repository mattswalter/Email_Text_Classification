# CISC684_Text_Classification
## By: Eric Allen, Matthew Walter, Murugesan Somasundaram
Requires Python 3.6

Libraries: os, sys, re, math, glob, collections, pandas, numpy,

Summary: Implement and evaluate Naive Bayes, Perceptron, and Logistic Regression for text classification.

Requires 5-7 additional arguments:

python main.py <program> <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <OPTIONAL: lambda> <OPTIONAL: iterations>

##### program: BAYES | MCAP | PERCEPTRON
##### path_ham_train: Directory Path
##### path_spam_train: Directory Path
##### path_ham_test: Directory Path
##### path_spam_test: Directory Path
##### lambda: Float Value (0.01 or 0.1 is recommended)
##### iterations: Integer Value

## To Run
  > python main.py BAYES <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test>
  
  > python main.py MCAP <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <lambda> <iterations>
 
  > python main.py PERCEPTRON <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <iterations>

## Example Run
  > python main.py BAYES hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/

  > python main.py MCAP hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/ 0.01 100

  > python main.py PERCEPTRON hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/ 1000
