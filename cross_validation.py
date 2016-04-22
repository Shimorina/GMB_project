import os
from random import shuffle
from random import seed
from collections import defaultdict


# 97 535 pairs
# separate them into 5 folds: 19 507 pairs each (= 20%).
# training data: 80%, test data: 20%.
# create 5 folders: fold1, fold2, fold3, etc

path = '/home/anastasia/Documents/GMB_crfs/syntax_refined/CRF_tests/'


def build_data_crf():
    global path
    label_distr = defaultdict(int)
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
                # count labels
                if line != '\n':
                    label = line.split('\t')[9].strip()
                    label_distr[label] += 1
            if counterline % 4 == 0:
                corpus_pairs.append(pair)
                pair = ''
            counterline += 1
        if pair != '':
            corpus_pairs.append(pair)

    print('Number of pairs: {}'.format(len(corpus_pairs)))
    for k in sorted(label_distr):
        print(k, label_distr[k])
    seed(1)
    shuffle(corpus_pairs)

    # create ten folds with training and test data
    for fold in range(1, 6):
        fold_dir = path + 'fold' + str(fold) + '/'
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        start_split = (fold-1) * 19507
        end_split = fold * 19507
        with open(fold_dir + 'training_pair_fold' + str(fold) + '.txt', 'w+') as f_train:
            f_train.write(''.join(corpus_pairs[:start_split]))
            f_train.write(''.join(corpus_pairs[end_split:]))
        with open(fold_dir + 'testing_pair_fold' + str(fold) + '.txt', 'w+') as f_test:
            f_test.write(''.join(corpus_pairs[start_split:end_split]))

        print('writing fold number {}...'.format(fold))
        print('writing testing file from {} to {} (not included)'.format(start_split, end_split))


# 61 920 sequences of different size (from 1 to 11) out of 62 010 sentences from the GMB as stated in their readme
#  split into 5 folds with 12 384 sequences each


def build_data_crf_sequences():
    global path
    ev_counter = 0
    max_event_number = 0  # find the maximum number of events in sequences
    label_distr = defaultdict(int)
    args_distr = defaultdict(int)  # count the number where sentences have common arguments
    x_count = False
    y_count = False
    z_count = False
    w_count = False
    # create a list containing all the pairs
    with open('training_data_sequences_all_refined.txt', 'r') as f:
        next(f)  # skip first line with a title
        corpus_sequences = []  # every element is a sequence
        seq = ''
        # i2003 ---> i2007 from p94/d0589
        # 1	-	-	-	1	-	-	-	-	S[dcl]\NP
        # 1	-	-	-	-	-	-	-	-	S[ng]\NP
        for line in f:
            # sentence is finished
            if line == '\n':
                seq += line
                corpus_sequences.append(seq)
                seq = ''
                if ev_counter > max_event_number:
                    max_event_number = ev_counter
                ev_counter = 0
                x_count = False
                y_count = False
                z_count = False
                w_count = False
                continue
            if line[0] != '#':  # skip lines with comments
                seq += line
                ev_counter += 1
                *features, label = line.split('\t')
                label = label.strip()
                label_distr[label] += 1
                if 'X' in features and x_count is False:
                    args_distr['X'] += 1
                    x_count = True
                elif 'Y' in features and y_count is False:
                    args_distr['Y'] += 1
                    y_count = True
                elif 'Z' in features and z_count is False:
                    args_distr['Z'] += 1
                    z_count = True
                elif 'W' in features and w_count is False:
                    args_distr['W'] += 1
                    w_count = True

        if seq != '':
            corpus_sequences.append(seq)
    print('Number of sequences: {}'.format(len(corpus_sequences)))
    print('Maximum number of events in a sequence: {}'.format(max_event_number))
    out = ''
    for k in sorted(label_distr, key=label_distr.get, reverse=True):
        percentage = round(label_distr[k] / (sum(label_distr.values()) / 100), 4)
        out += '{} {} ({} %) \n'.format(k, label_distr[k], percentage)
        out += '\n'
    out += '=== Distribution of common arguments ===\n'
    for k in sorted(args_distr, key=args_distr.get, reverse=True):
        out += 'Argument {} was found in {} sentences\n'.format(k, args_distr[k])
        out += '\n'
    with open(path + 'label_distr_refined.txt', 'w+') as f:
        f.write(out)

    seed(1)
    shuffle(corpus_sequences)

    # create ten folds with training and test data
    for fold in range(1, 6):
        fold_dir = path + 'fold' + str(fold) + '/'
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        start_split = (fold-1) * 12384
        end_split = fold * 12384
        with open(fold_dir + 'training_seq_fold' + str(fold) + '.txt', 'w+') as f_train:
            f_train.write(''.join(corpus_sequences[:start_split]))
            f_train.write(''.join(corpus_sequences[end_split:]))
        with open(fold_dir + 'testing_seq_fold' + str(fold) + '.txt', 'w+') as f_test:
            f_test.write(''.join(corpus_sequences[start_split:end_split]))

        print('writing fold number {}...'.format(fold))
        print('writing testing file from {} to {} (not included)'.format(start_split, end_split))


def calculate_precision(file_dir):
    res_path = '/home/anastasia/Documents/GMB_crfs/syntax_refined/CRF_results/template3/'
    # res_path = 'C:/Users/Anastassie/Dropbox/Loria/GMB/CRF_results/template3/'
    accuracy = []
    out = ''
    prec_recall_folds = {}  # label: [Precision-fold1, Recall-fold1, P-fold2, R-fold2, etc]
    for fold in range(1, 6):
        prec_recall = {}  # label: [tp, fp, fn] for each fold
        dissimilar_dict = defaultdict(int)
        with open(res_path + file_dir + str(fold) + '.txt', 'r') as f:
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
                    if initial not in prec_recall:
                        prec_recall[initial] = [0, 0, 0]
                    if predicted not in prec_recall:
                        prec_recall[predicted] = [0, 0, 0]
                    if initial == predicted:  # true positives
                        prec_recall[initial][0] += 1
                    elif initial != predicted:
                        prec_recall[initial][1] += 1  # false positives
                        prec_recall[predicted][2] += 1  # false negatives
        fold_precision = 100 - (dissimilar_count / (all_occur/100))
        print('Accuracy: {}'.format(fold_precision))
        accuracy.append(fold_precision)
        out += str(fold) + ' fold\n'
        for k in sorted(dissimilar_dict, key=dissimilar_dict.get, reverse=True):
            percent = round(dissimilar_dict[k] / (dissimilar_count / 100), 4)
            out += '{}\t{} ({} %) \n'.format(k, dissimilar_dict[k], percent)
        out += '\n\n'
        for k in sorted(prec_recall, key=prec_recall.get, reverse=True):
            tp = prec_recall[k][0]
            fp = prec_recall[k][1]
            fn = prec_recall[k][2]
            precision = round(tp / (tp + fp), 4)
            if tp == 0 and fn == 0:
                recall = 0
            else:
                recall = round(tp / (tp + fn), 4)
            out += '{} precision: {} recall: {} \n'.format(k, precision, recall)
            if k not in prec_recall_folds:
                prec_recall_folds[k] = [precision, recall]
            else:
                prec_recall_folds[k] += [precision, recall]
        out += '\n\n'
    out += '====== Averaged precision and recall over five folds ======\n'
    for label in sorted(prec_recall_folds, key=prec_recall_folds.get, reverse=True):
        av_precision = round(sum(prec_recall_folds[label][0::2])/5, 4)  # even elements are precisions for each fold
        av_recall = round(sum(prec_recall_folds[label][1::2])/5, 4)  # odd elements are recalls for each fold
        out += '{} precision: {} recall: {} \n'.format(label, av_precision, av_recall)
    print('Averaged accuracy for five folds: {}'.format(sum(accuracy) / len(accuracy)))
    with open(res_path + 'crf_errors_' + file_dir + '.txt', 'w+') as f:
        f.write(out)


seq_dir = 'out_seq_fold'
pair_dir = 'out_pair_fold'
# calculate_precision(seq_dir)
# calculate_precision(pair_dir)
build_data_crf()
# build_data_crf_sequences()
