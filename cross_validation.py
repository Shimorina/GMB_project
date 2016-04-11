import os
from random import shuffle
from random import seed
from collections import defaultdict


# 97 539 pairs
# separate them into 5 folds: 19 507 pairs each (= 20%).
# training data: 80%, test data: 20%.
# create 5 folders: fold1, fold2, fold3, etc

path = '/home/anastasia/Documents/CRF_tests'


def build_data_crf():
    global path
    # create a list containing all the pairs
    with open('training_data_pairs_all.txt', 'r') as f:
        next(f)  # skip first line with a title
        corpus_pairs = []  # every element is a pair
        pair = ''
        counterline = 1
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
        start_split = (fold-1) * 19507
        end_split = fold * 19507
        with open(fold_dir + '/training_pair_fold' + str(fold) + '.txt', 'w+') as f_train:
            f_train.write(''.join(corpus_pairs[:start_split]))
            f_train.write(''.join(corpus_pairs[end_split:]))
        with open(fold_dir + '/testing_pair_fold' + str(fold) + '.txt', 'w+') as f_test:
            f_test.write(''.join(corpus_pairs[start_split:end_split]))

        print('writing fold number {}...'.format(fold))
        print('writing testing file from {} to {} (not included)'.format(start_split, end_split))


# 61 362 sequences of different size (from 1 to 11)
#


def build_data_crf_sequences():
    global path
    ev_counter = 0
    max_event_number = 0  # find the maximum number of events in sequences
    label_distr = defaultdict(int)
    # create a list containing all the pairs
    with open('training_data_sequences_all.txt', 'r') as f:
        next(f)  # skip first line with a title
        corpus_sequences = []  # every element is a sequence
        seq = ''
        # i2003 ---> i2007 from p94/d0589
        # 1	-	-	-	1	-	-	-	-	S[dcl]\NP
        # 1	-	-	-	-	-	-	-	-	S[ng]\NP
        for line in f:
            if line == '\n':
                seq += line
                corpus_sequences.append(seq)
                seq = ''
                if ev_counter > max_event_number:
                    max_event_number = ev_counter
                ev_counter = 0
                continue
            if line[0] != '#':  # skip lines with comments
                seq += line
                ev_counter += 1
                label = line.split('\t')[9].strip()
                label_distr[label] += 1
        if seq != '':
            corpus_sequences.append(seq)
    print(len(corpus_sequences))
    print(max_event_number)
    out = ''
    for k in sorted(label_distr, key=label_distr.get, reverse=True):
        percentage = round(label_distr[k] / (sum(label_distr.values()) / 100), 4)
        out += '{} {} ({} %) \n'.format(k, label_distr[k], percentage)
        out += '\n'
    with open(path + '_label_distr.txt', 'w+') as f:
        f.write(out)

    seed(1)
    shuffle(corpus_sequences)

    # create ten folds with training and test data
    '''for fold in range(1, 6):
        fold_dir = path + '/fold' + str(fold)
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        start_split = (fold-1) * 12272
        end_split = fold * 12272
        with open(fold_dir + '/training_seq_fold' + str(fold) + '.txt', 'w+') as f_train:
            f_train.write(''.join(corpus_sequences[:start_split]))
            f_train.write(''.join(corpus_sequences[end_split:]))
        with open(fold_dir + '/testing_seq_fold' + str(fold) + '.txt', 'w+') as f_test:
            f_test.write(''.join(corpus_sequences[start_split:end_split]))

        print('writing fold number {}...'.format(fold))
        print('writing testing file from {} to {} (not included)'.format(start_split, end_split))'''


def calculate_precision(file_dir):
    path = '/home/anastasia/Documents/CRF_results/template1/'
    precision_all = []
    out = ''
    prec_recall = {}  # label: [tp, fp, fn]
    for fold in range(1, 6):
        dissimilar_dict = defaultdict(int)
        with open(path + file_dir + str(fold) + '.txt', 'r') as f:
            dissimilar_count = 0
            all_occur = 0
            for line in f:
                if line != '\n':
                    all_occur += 1
                    el = line.split('\t')
                    initial = el[9]
                    predicted = el[10].strip()
                    if initial != predicted:
                        dissimilar_count += 1
                        error_name = initial + ' --> ' + predicted
                        dissimilar_dict[error_name] += 1
                    # add to the dict of true positives, false positives and false negatives
                    '''if initial == predicted:  # true positives
                        prec_recall.setdefault(initial, []).update()
                    elif initial != predicted:
                        prec_recall.setdefault(initial, []).update()  # false positives
                        prec_recall.setdefault(predicted, []).update()  # false negatives'''
        fold_precision = 100 - (dissimilar_count / (all_occur/100))
        print('Accuracy: {}'.format(fold_precision))
        precision_all.append(fold_precision)
        out += str(fold) + ' fold\n'
        for k in sorted(dissimilar_dict, key=dissimilar_dict.get, reverse=True):
            percent = round(dissimilar_dict[k] / (dissimilar_count / 100), 4)
            out += '{}\t{} ({} %) \n'.format(k, dissimilar_dict[k], percent)
        out += '\n'
    print('Averaged accuracy for five folds: {}'.format(sum(precision_all) / len(precision_all)))
    with open(path + 'crf_errors_' + file_dir + '.txt', 'w+') as f:
        f.write(out)


seq_dir = 'out_seq_fold'
pair_dir = 'out_pair_fold'
calculate_precision(seq_dir)
# calculate_precision(pair_dir)
# build_data_crf()
# build_data_crf_sequences()
