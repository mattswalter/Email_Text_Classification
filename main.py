# =============================================================================
# #
# #  Created on Sun Sep 29 10:16:43 2019
# #
# #  CISC684: Group 2
# #
# #  @author: Eric Allen <allenea@udel.edu>
# #           Matthew Walter <mswalter@udel.edu>        
# #           Murugesan Somasundaram <smuruges@udel.edu>
# #
# #  Command Line Execution:
# #
# #      python main.py BAYES <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test>
# #      python main.py BAYES hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/
# #      python main.py BAYES hw2datasets/dataset2/train/ham/ hw2datasets/dataset2/train/spam/ hw2datasets/dataset2/test/ham/ hw2datasets/dataset2/test/spam/
# #      python main.py BAYES hw2datasets/dataset3/train/ham/ hw2datasets/dataset3/train/spam/ hw2datasets/dataset3/test/ham/ hw2datasets/dataset3/test/spam/
# #
# #
# #      python main.py MCAP <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <lambda> <iterations>
# #      python main.py MCAP hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/ 0.01 100
# #      python main.py MCAP hw2datasets/dataset2/train/ham/ hw2datasets/dataset2/train/spam/ hw2datasets/dataset2/test/ham/ hw2datasets/dataset2/test/spam/ 0.01 100
# #      python main.py MCAP hw2datasets/dataset3/train/ham/ hw2datasets/dataset3/train/spam/ hw2datasets/dataset3/test/ham/ hw2datasets/dataset3/test/spam/ 0.01 100
# #
# #
# #      python main.py PERCEPTRON <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <iterations>
# #      python main.py PERCEPTRON hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/ 1000
# #      python main.py PERCEPTRON hw2datasets/dataset2/train/ham/ hw2datasets/dataset2/train/spam/ hw2datasets/dataset2/test/ham/ hw2datasets/dataset2/test/spam/ 1000
# #      python main.py PERCEPTRON hw2datasets/dataset3/train/ham/ hw2datasets/dataset3/train/spam/ hw2datasets/dataset3/test/ham/ hw2datasets/dataset3/test/spam/ 1000
# #
# =============================================================================
#IMPORTS
from __future__ import print_function
import os, sys           
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bayes import bayes
from src.LogisticRegression import MCAP
from src.perceptron import PERCEPTRON


def main():
    if len(sys.argv) < 6:
        print("INVALID INPUT")
        print("python main.py <program> <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <OPTIONAL: lambda> <OPTIONAL: iterations>")
        sys.exit(0)
    else:
        if "BAYES" == sys.argv[1].upper():
            if len(sys.argv) < 6:
                print("INVALID INPUT")
                print("<PROGRAM> (OPTIONS: BAYES/MCAP/PERCEPTRON)")
                print("python main.py <program> <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test>")
                sys.exit(0)
            else:
                bayes(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        elif "MCAP" == sys.argv[1].upper():
            if len(sys.argv) < 8:
                print("INVALID INPUT")
                print("python main.py MCAP <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <lambda> <iterations>")
                sys.exit(0)
            else:
                MCAP(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
        elif "PERCEPTRON" == sys.argv[1].upper():
            if len(sys.argv) < 7:
                print("INVALID INPUT")
                print("python main.py MCAP <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <iterations>")
                sys.exit(0)
            else:
                PERCEPTRON(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
        else:
            print("INVALID <PROGRAM> (OPTIONS: BAYES/MCAP/PERCEPTRON)")
            print("python main.py <program> <path_ham_train> <path_spam_train> <path_ham_test> <path_spam_test> <OPTIONAL: lambda> <OPTIONAL: iterations>")
            sys.exit(0)
        
        
if __name__ == "__main__":  
    main()
    
    
    
    
"""RESULTS
#BAYES
(base) wifi-roaming-128-4-71-6:Text_Classification_Group2 ericallen$ python main.py BAYES hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/
Correctly Guessed: 448
Inorrectly Guessed: 30
Overall Accuracy: 0.9372384937238494

(base) wifi-roaming-128-4-71-6:Text_Classification_Group2 ericallen$ python main.py BAYES hw2datasets/dataset2/train/ham/ hw2datasets/dataset2/train/spam/ hw2datasets/dataset2/test/ham/ hw2datasets/dataset2/test/spam/
Correctly Guessed: 427
Inorrectly Guessed: 29
Overall Accuracy: 0.9364035087719298

(base) wifi-roaming-128-4-71-6:Text_Classification_Group2 ericallen$ python main.py BAYES hw2datasets/dataset3/train/ham/ hw2datasets/dataset3/train/spam/ hw2datasets/dataset3/test/ham/ hw2datasets/dataset3/test/spam/
Correctly Guessed: 471
Inorrectly Guessed: 72
Overall Accuracy: 0.8674033149171271



#MCAP
(base) wifi-roaming-128-4-71-6:Text_Classification_Group2 ericallen$ python main.py MCAP hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/ 0.01 100
Correctly Guessed: 432
Total Guessed: 478
Overall Accuracy: 90.3765690376569

(base) wifi-roaming-128-4-71-6:Text_Classification_Group2 ericallen$ python main.py MCAP hw2datasets/dataset2/train/ham/ hw2datasets/dataset2/train/spam/ hw2datasets/dataset2/test/ham/ hw2datasets/dataset2/test/spam/ 0.01 100
Correctly Guessed: 420
Total Guessed: 456
Overall Accuracy: 92.10526315789474

(base) wifi-roaming-128-4-71-6:Text_Classification_Group2 ericallen$ python main.py MCAP hw2datasets/dataset3/train/ham/ hw2datasets/dataset3/train/spam/ hw2datasets/dataset3/test/ham/ hw2datasets/dataset3/test/spam/ 0.01 100
Correctly Guessed: 510
Total Guessed: 543
Overall Accuracy: 93.92265193370166



#PERCEPTRON
(base) wifi-roaming-128-4-71-6:Text_Classification_Group2 ericallen$ python main.py PERCEPTRON hw2datasets/dataset1/train/ham/ hw2datasets/dataset1/train/spam/ hw2datasets/dataset1/test/ham/ hw2datasets/dataset1/test/spam/ 1000
Correctly Guessed: 436
Total Guessed: 478
Overall Accuracy: 91.21338912133892

(base) wifi-roaming-128-4-71-6:Text_Classification_Group2 ericallen$ python main.py PERCEPTRON hw2datasets/dataset2/train/ham/ hw2datasets/dataset2/train/spam/ hw2datasets/dataset2/test/ham/ hw2datasets/dataset2/test/spam/ 1000
Correctly Guessed: 417
Total Guessed: 456
Overall Accuracy: 91.44736842105263

(base) wifi-roaming-128-4-71-6:Text_Classification_Group2 ericallen$ python main.py PERCEPTRON hw2datasets/dataset3/train/ham/ hw2datasets/dataset3/train/spam/ hw2datasets/dataset3/test/ham/ hw2datasets/dataset3/test/spam/ 1000
Correctly Guessed: 512
Total Guessed: 543
Overall Accuracy: 94.29097605893186
"""