import os
from random import shuffle
from random import seed


# 98 087 pairs
# separate them into 5 folds: 19 617 pairs each (= 20%).
# training data: 80%, test data: 20%.
# create 5 folders: fold1, fold2, fold3, etc

path = '/home/anastasia/Documents/CRF_tests'

def build_data_CRF():
    global path
    # create a list containing all the pairs
    with open('training_data_pairs_all.txt', 'r') as f:
        next(f)  # skip first line with a title
        corpus_pairs = []  # every element is a pair
        pair = ''
        counterline = 1
        fold = 1
        # i2003 ---> i2007 from p94/d0589
        # 1	-	-	-	1	-	-	-	-	S[dcl]\NP
        # 1	-	-	-	-	-	-	-	-	S[ng]\NP
        for line in f:
            if line[0] != '#':  # skip lines with comments
                pair += line
            if counterline % 4 == 0:
                corpus_pairs.append(pair)
                pair = ''
            counterline += 1
        if pair != '':
            corpus_pairs.append(pair)

    print(len(corpus_pairs))
    seed(1)
    shuffle(corpus_pairs)

    # create ten folds with training and test data
    for fold in range(1, 6):
        fold_dir = path + '/fold' + str(fold)
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        start_split = (fold-1) * 19617
        end_split = fold * 19617
        with open(fold_dir + '/training_fold' + str(fold) + '.txt', 'w+') as f_train:
            f_train.write(''.join(corpus_pairs[:start_split]))
            f_train.write(''.join(corpus_pairs[end_split:]))
        with open(fold_dir + '/testing_fold' + str(fold) + '.txt', 'w+') as f_test:
            f_test.write(''.join(corpus_pairs[start_split:end_split]))

        print('writing fold number {}...'.format(fold))
        print('writing testing file from {} to {} (not included)'.format(start_split, end_split))


def calculate_precision():
    path = '/home/anastasia/Documents/CRF_results/'
    precision_all = []
    for fold in range(1, 4):
        with open(path + 'out_fold' + str(fold) + '.txt', 'r') as f:
            dissimilar_count = 0
            all_occur = 0
            for line in f:
                if line != '\n':
                    all_occur += 1
                    el = line.split('\t')
                    if el[9] != el[10].strip():
                        dissimilar_count += 1
        # print(all_occur)
        # print(dissimilar_count)
        fold_precision = 100 - (dissimilar_count / (all_occur/100))
        print('Precision: {}'.format(fold_precision))
        precision_all.append(fold_precision)

    print('Averaged precision on five folds: {}'.format(sum(precision_all) / len(precision_all)))


calculate_precision()






