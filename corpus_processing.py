import os
import csv
import itertools
import xml.etree.ElementTree as ETree
from collections import defaultdict


GMB_path = '/home/anastasia/Documents/the_GMB_corpus/gmb-2.2.0/data/'
# GMB_path = '/home/anastasia/Documents/the_GMB_corpus/gmb-2.2.0/data_t/'
# GMB_path = '/home/anastasia/Documents/the_GMB_corpus/gmb-2.2.0/data_test/'
# GMB_path = 'C:/Users/Anastassie/Documents/Loria/GMB/gmb-2.2.0/data_test'
roles = ['agent', 'asset', 'attribute', 'beneficiary', 'cause', 'co-agent', 'co-theme', 'destination', 'extent',
         'experiencer', 'frequency', 'goal', 'initial_location', 'instrument', 'location', 'manner', 'material',
         'patient', 'path', 'pivot', 'product', 'recipient', 'result', 'source', 'stimulus', 'time', 'topic', 'theme',
         'trajectory', 'value', 'agent-1', 'asset-1', 'attribute-1', 'beneficiary-1', 'cause-1', 'co-agent-1',
         'co-theme-1', 'destination-1', 'extent-1', 'experiencer-1', 'frequency-1', 'goal-1', 'initial_location-1',
         'instrument-1', 'location-1', 'manner-1', 'material-1', 'patient-1', 'path-1', 'pivot-1', 'product-1',
         'recipient-1', 'result-1', 'source-1', 'stimulus-1', 'time-1', 'topic-1', 'theme-1', 'trajectory-1', 'value-1']
ccg_cats = defaultdict(int)
roles_dict = {}
sems_synt = {}
pred_which_count = 0
word_which_count = 0


def read_corpus(gmb_path):
    """
    Read the directory with the Groningen Meaning bank corpus. Call the function drg_mining for each file.
    Create csv files where the data are to be written and files with stats.
    :param gmb_path: path to the GMB corpus
    :return:
    """
    count = 0
    global roles_dict
    global ccg_cats
    global sems_synt
    with open('events_all.csv', 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Path', 'Token', 'Event', 'Offset', 'Event Predicate', 'Thematic roles',
                            'Temporalities', 'Other Semantics [surface form if any]',
                            'Attributes ("arg" edges) [surface form if any]', 'Entities',
                            'Propositions', 'Surfaces', 'Function words', 'Referents',
                            'Sentence', 'Guess Offset'])
    with open('ccg_categories_all.csv', 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Path', 'Token', 'Offset', 'CCG category', 'Sentence', 'Guess Offset'])
    with open('training_data_pairs_all.txt', 'w+') as f:
        f.write('#Agent\tPatient\tAgent-1\tPatient-1\tTheme\tTheme-1\tRecipient\tRecipient-1\tTopic\tSyntacticLabel\n')
    with open('training_data.txt', 'w+') as f:
        f.write('#Agent\tPatient\tAgent-1\tPatient-1\tTheme\tTheme-1\tRecipient\tRecipient-1\tTopic\tSyntacticLabel\n')
    for partition in os.listdir(gmb_path):
        partition_path = gmb_path + '/' + partition  # get path of each partition
        for entry in os.listdir(partition_path):
            curr_directory = partition_path + '/' + entry + '/'
            count += 1
            drg_mining(curr_directory)
    print('Number of docs in GMB: {}'.format(count))
    with open('stats.txt', 'w+') as f:
        for roles in sorted(roles_dict, key=roles_dict.get, reverse=True):
            f.write(str(roles) + '\t' + str(roles_dict[roles]) + '\n')
        f.write('\n=====CCG Categories====\n\n')
        for cats in sorted(ccg_cats, key=ccg_cats.get, reverse=True):
            f.write(str(cats) + '\t' + str(ccg_cats[cats]) + '\n')
    out = ''
    with open('semantics_vs_syntax.txt', 'w+') as f:
        # sort the sem_synt dict by the sum of values of its embedded dicts
        sort_func = lambda abr_roles: sum(value for value in sems_synt[abr_roles].values())
        for abr_roles in sorted(sems_synt, key=lambda abr_roles: sum(value for value in sems_synt[abr_roles].values()), reverse=True):
            out += str(abr_roles) + '\t' + str(sort_func(abr_roles)) + '\n=========\n'
            for ccg in sorted(sems_synt[abr_roles], key=sems_synt[abr_roles].get, reverse=True):
                out += ccg + '\t' + str(sems_synt[abr_roles][ccg]) + '\n'
            out += '\n\n'
        f.write(out)


def drg_mining(file_path):
    """
    Read DRG file. For each event call the function event_relation and get its offset and ccg category.
    Write extracted data to csv files.
    Construct two types of the training data for CRFs.
    :param file_path: path to a file in the GMB corpus
    :return:
    """
    # Read en.tags file and build a list of sentences
    sent = []
    sentence = []
    file_train_set = {}
    global roles_dict
    global ccg_cats
    global sems_synt
    global word_which_count
    global pred_which_count
    with open(file_path+'en.tags', 'r') as f:
        for line in f:
            token = line.split('\t')[0]
            if token != '\n':
                sentence += [token]
            else:
                sent += [sentence]
                sentence = []
        sent += [sentence]  # file does not end with an empty line

    # Read drg file
    drg_tuples = []  # tuples of DRG
    with open(file_path+'en.drg', 'r') as f:
        next(f)  # skip first three lines
        next(f)
        next(f)
        for line in f:
            if line != '\n':
                drg_tuples.append(line.rstrip().split())

    # Find events (by searching subtype event)
    events = []
    for dtuple in drg_tuples:
        if dtuple[1] == 'event':
            events.append(dtuple[2])  # k3:p1 event c41:open:1 0

    for event in events:
        # Get a surface form of the event
        for dtuple in drg_tuples:
            if dtuple[0] == event:
                if dtuple[1] == 'instance':
                    token = dtuple[5]  # c52:spot:1 instance k3:p1:e5 3 [ spotting ]
                    event_id = dtuple[2]
                    pure_event_id = event_id.split(':')[-1]
                    predicate = dtuple[0].split(':')[1]
                    predicate_arg = predicate + '(' + pure_event_id + ')'
                else:
                    print('wrong format')
        if token == ']':
            if predicate == 'event':
                token = 'EVENT'
            else:
                token = 'EllipticalEvent'

        # Get event relations
        them_roles_smart, them_roles, temporalities, semantics, attributes,\
        instances, surfaces, propositions, connectives, connectives2 = event_relation(drg_tuples, event_id)
        # Get offset of the event
        offset, guess = get_sentences(file_path, pure_event_id, predicate)  # i16014, no
        # Get sentence with the event in question
        if offset != 'Not available':
            target_sent = sent[int(offset[1:-3]) - 1]
        else:
            target_sent = 'Unknown'

        fpath_short = file_path.split('/')[-3] + '/' + file_path.split('/')[-2]
        with open('events_all.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([fpath_short, token, event_id, offset, predicate_arg, ', '.join(them_roles), ', '.join(temporalities), ', '.join(semantics), ', '.join(attributes), ', '.join(instances), ', '.join(propositions), '||'.join(surfaces), ' '.join(connectives), ' '.join(connectives2), ' '.join(target_sent), guess])
        # Get ccg categories
        ccg_category = ccg_categories(token, offset, file_path)
        with open('ccg_categories_all.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([fpath_short, token, offset, ccg_category, ' '.join(target_sent), guess])
        # calculate stats of thematic roles and ccg categories
        roles_dict, ccg_cats, abr_roles = calculate_stats(them_roles, roles_dict, ccg_category, ccg_cats)
        # abridge semantic roles
        if abr_roles not in sems_synt:
            sems_synt[abr_roles] = defaultdict(int)
            sems_synt[abr_roles][ccg_category] = 1
        else:
            sems_synt[abr_roles][ccg_category] += 1
        # calculate the match between words in sents and predicates in semantics
        for pred in semantics:
            if 'which' in pred:
                pred_which_count += 1
        if 'which' in target_sent or 'Which' in target_sent:
            word_which_count += 1
        # collect training data for each file
        ccg_cat_norm = normalise_ccg_cat(ccg_category)
        # put all events of one file to the dict having the structure: i450 : [[agent:x1, patient:x56, recipient:x4], NP/S]
        file_train_set[offset] = [them_roles_smart, ccg_cat_norm]

    # write training data to file
    '''with open('training_data_pairs_all.txt', 'a') as f:
        placeholder_dict = ['agent', 'patient', 'agent-1', 'patient-1', 'theme', 'theme-1', 'recipient', 'recipient-1', 'topic']
        placeholder = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
        replacements = ['X', 'Y', 'Z', 'W', 'U']
        out = ''
        events = sorted(file_train_set.keys())
        # for each pair of events, we will extract features and set them to 1 or to X if roles of two events have the argument in common
        for i in range(len(events) - 1):
            event = events[i]
            next_event = events[i + 1]
            if event[1] != next_event[1]:  # events must be in one sentence! (the second letter of offset is equal to the sent number)
                continue
            out += '#' + event + ' ---> ' + next_event + ' from ' + fpath_short + '\n'
            features, label = file_train_set[event]
            features2, label2 = file_train_set[next_event]
            # check if two sets of features have same arguments of thematic roles
            arguments = [feat.split(':')[1] for feat in sorted(features)]
            arguments2 = [feat.split(':')[1] for feat in sorted(features2)]
            common_elements = [val for val in arguments if val in arguments2]
            # remove duplicates; we can't use sets as the order is important
            common_elements_no_dup = []
            for el in common_elements:
                if el not in common_elements_no_dup:
                    common_elements_no_dup.append(el)
            # replace common args with X, Y, Z, etc in a placeholder
            if common_elements_no_dup:
                j = 0  # iterate over replacements: X, Y, Z, etc
                placeholder_with_x_second = list(placeholder)
                placeholder_with_x_first = list(placeholder)
                for element in common_elements_no_dup:
                    # find the role of the common element and replace it with X
                    # do it for the first event in a pair
                    for feat in sorted(features):
                        role, argument = feat.split(':')
                        if element == argument:
                            role_index = placeholder_dict.index(role)
                            placeholder_with_x_first[role_index] = replacements[j]
                    # for the second event in a pair
                    for feat in sorted(features2):
                        role, argument = feat.split(':')
                        if element == argument:
                            role_index = placeholder_dict.index(role)
                            placeholder_with_x_second[role_index] = replacements[j]
                    j += 1
            else:
                placeholder_with_x_first = list(placeholder)
                placeholder_with_x_second = list(placeholder)

            # replace all other args with 1 except for X, Y, etc
            for feature in features:
                role, argument = feature.split(':')
                role_index = placeholder_dict.index(role)
                if placeholder_with_x_first[role_index] not in replacements:
                    placeholder_with_x_first[role_index] = '1'
            # write to output the first event
            out += ('\t').join(placeholder_with_x_first) + '\t' + label + '\n'

            # replace all other args with 1 except for X, Y, etc
            for feature in features2:
                role, argument = feature.split(':')
                role_index = placeholder_dict.index(role)
                if placeholder_with_x_second[role_index] not in replacements:
                    placeholder_with_x_second[role_index] = '1'
            # write to output the second event
            out += '\t'.join(placeholder_with_x_second) + '\t' + label2 + '\n\n'

        f.write(out)'''

    # write training data for CRFs to file
    with open('training_data.txt', 'a') as f:  ## i450 : [[agent:x1, patient:x56, recipient:x4], NP/S]
        placeholder_dict = ['agent', 'patient', 'agent-1', 'patient-1', 'theme', 'theme-1', 'recipient', 'recipient-1', 'topic']
        placeholder = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
        sent_placeholder = []
        replacements = ['X', 'Y', 'Z', 'W', 'U']
        out2 = ''
        # group events by sentences
        events = sorted(file_train_set.keys())
        sents = []   # contains dicts of events for each sentence: [{i100:[[roles], label], i109:[[roles], label], ...}, {i203:...}, {}, ...]
        for event in events:
            sent_number = int(event[1])  # i3010
            if len(sents) < sent_number:
                sents.insert(sent_number-1, {})
                sents[sent_number - 1][event] = file_train_set[event]
            else:
                sents[sent_number - 1][event] = file_train_set[event]
        # for each sentence, extract features and set them to 1 or to X, Y, etc if roles have the argument in common
        for sent in sents:
            out2 += '#' + '--->'.join(sorted(sent.keys())) + ' from ' + fpath_short + '\n'
            sent_args = []  # contains lists of args for each event
            for event in sent:
                features, label = file_train_set[event]
                arguments = [feat.split(':')[1] for feat in features]
                sent_args.append(arguments)
            # for each pair of events, check if they have elements in common
            common_elements = []
            for combination in itertools.combinations(sent_args, len(placeholder_dict)):
                common_elements.append(set(combination))

            # replace common args with X, Y, Z, etc in a placeholder
            if common_elements:
                j = 0  # iterate over replacements: X, Y, Z, etc
                for element in common_elements:
                    # find the role of the common element and replace it with X
                    # do it for the first event in a pair
                    for feat in features:
                        role, argument = feat.split(':')
                        if element == argument:
                            role_index = placeholder_dict.index(role)
                            placeholder_with_x_first = list(placeholder)
                            placeholder_with_x_first[role_index] = replacements[j]
                    # for the second event in a pair
                    for feat in features:
                        role, argument = feat.split(':')
                        if element == argument:
                            role_index = placeholder_dict.index(role)
                            placeholder_with_x_second = list(placeholder)
                            placeholder_with_x_second[role_index] = replacements[j]
                    j += 1
            else:
                placeholder_with_x_first = list(placeholder)
                placeholder_with_x_second = list(placeholder)

            # replace all other args with 1 except for X
            for feature in features:
                role, argument = feature.split(':')
                role_index = placeholder_dict.index(role)
                if placeholder_with_x_first[role_index] not in replacements:
                    placeholder_with_x_first[role_index] = '1'
            # write to output the first event
            out2 += ('\t').join(placeholder_with_x_first) + '\t' + label + '\n'

            # replace all other args with 1 except for X
            for feature in features:
                role, argument = feature.split(':')
                role_index = placeholder_dict.index(role)
                if placeholder_with_x_second[role_index] not in replacements:
                    placeholder_with_x_second[role_index] = '1'
            # write to output the second event
            out2 += '\t'.join(placeholder_with_x_second) + '\t' + label + '\n\n'

        f.write(out2)


semtypes = set()
edges = set()


def event_relation(drg_tuples, event_id):
    """
    Walk the DR graph and extract all event relations with their attributes.
    :param drg_tuples: the DRG -- a list of tuples where two nodes are connected with an edge
    :param event_id: id of the event in a DRG tuple, e.g. k3:p1:e5
    :return:
    """
    current_triple = ['REL', 'INT', 'EXT']  # contain conditions ['agent', 'e1', 'x4'] : e.g. "agent (e1, x4)", "temp_included(e20, t13)"
    them_roles = []
    them_roles_smart = []  # role:event_arg, e.g. agent-1:e9
    temporalities = []
    relations = []
    attributes = []
    instances = []
    surfaces = []
    propositions = []
    connectives = []
    connectives2 = []
    global semtypes
    global roles
    # c54:patient:1 int k3:p1:e5 4 [ ]
    for dtuple in drg_tuples:
        edges.add(dtuple[1])
        # Exclude referent and condition tuples ("k6 referent k6:e1") and argument tuples of type "instance"
        if dtuple[2] == event_id and dtuple[0].startswith('c') and dtuple[1] != 'instance':  # tuple should start with "c" and the edge is not of type "instance"
            sem_id = dtuple[0]
            argument = dtuple[1]
            # check if the relation is inverted, e.g. agent:-1
            if sem_id.split(':')[-1] == '-1':
                current_triple[0] = sem_id.split(':')[-2] + '-1'  # patient-1
            else:
                current_triple[0] = sem_id.split(':')[-2]  # patient
            if argument == 'int':
                current_triple[1] = event_id.split(':')[-1]  # e5
            elif argument == 'ext':
                current_triple[2] = event_id.split(':')[-1]  # e5
            elif argument == 'arg':  # c57:late:1   arg k2:e5   6   [   late  ]
                if dtuple[5] != ']':
                    attributes.append(sem_id.split(':')[-2] + '(' + event_id.split(':')[-1] + ') [' + dtuple[5] + ']')
                else:
                    attributes.append(sem_id.split(':')[-2] + '(' + event_id.split(':')[-1] + ') [EllipticalAttr]')
            elif argument == 'surface':
                surfaces.append(dtuple[5])

            # c54:patient:1 ext k3:p1:x16 0 [ ]
            for dtuple in drg_tuples:
                if dtuple[0] == sem_id:   # todo more fine-grained algorithm: int, ext, arg
                    # collect event ids in propositions if any
                    if 'p' in dtuple[2].split(':')[-1]:
                        propositions = get_propositions(drg_tuples, propositions, dtuple[2])

                    if argument == 'int' and dtuple[1] == 'ext':
                        current_triple[2] = dtuple[2].split(':')[-1]  # x16
                        inst_id = dtuple[2]
                        triple = current_triple[0] + '(' + current_triple[1] + ', ' + current_triple[2] + ')'
                        if current_triple[0] in roles:
                            them_roles.append(triple)
                            them_roles_smart.append(current_triple[0] + ':' + current_triple[2])
                        elif current_triple[0].startswith('temp'):
                            temporalities.append(triple)
                            temporalities = get_temporalities(drg_tuples, temporalities, current_triple[0], dtuple)
                        else:
                            # add surface form if exists
                            if dtuple[5] != ']':
                                triple += ' [' + dtuple[5] + ']'
                            relations.append(triple)  # triple is filled
                        semtypes.add(current_triple[0])

                        for dtuple in drg_tuples:
                            if dtuple[2] == inst_id:
                                inst_argument = inst_id.split(':')[-1]  # x16
                                if dtuple[1] == 'instance':  # c49:people:1 instance k3:p1:x16 2 [ people ]
                                    instances.append(dtuple[0].split(':')[1] + '(' + inst_argument + ')')  # get lemma "people"
                                elif dtuple[1] == 'arg':    # c19:nearly:1  arg k1:x7   2   [   nearly ]
                                    attributes.append(dtuple[0].split(':')[1] + '(' + inst_argument + ')')
                                elif dtuple[1] == 'surface':  # k8  surface k8:e17  2   [   and   ]
                                    surfaces.append(dtuple[5])

                                if dtuple[1] == 'referent' and dtuple[5] != ']':  # k2 referent k2:x11 1 [ which ]
                                    connectives.append('referent "' + dtuple[5] + '" (' + inst_argument + ')')

                    elif argument == 'ext' and dtuple[1] == 'int':
                        current_triple[1] = dtuple[2].split(':')[-1]  # x16
                        inst_id = dtuple[2]
                        triple = current_triple[0] + '(' + current_triple[1] + ', ' + current_triple[2] + ')'
                        if current_triple[0] in roles:
                            them_roles.append(triple)
                            them_roles_smart.append(current_triple[0] + ':' + current_triple[1])
                        elif current_triple[0].startswith('temp'):
                            temporalities.append(triple)
                            temporalities = get_temporalities(drg_tuples, temporalities, current_triple[0], dtuple)
                        else:
                            relations.append(triple)  # triple is filled
                        semtypes.add(current_triple[0])

                        # Search for equivs in drg and extract surface forms from referents if any
                        # c209:equality ext k76:x37
                        for dtuple in drg_tuples:
                            if dtuple[2] == inst_id and dtuple[1] == 'referent' and dtuple[5] != ']':  # k4:p3 referent k4:p3:x28 1 [ that ]
                                connectives2.append('referent "' + dtuple[5] + '" (' + inst_id.split(':')[-1] + ')')
                            eq_node = dtuple[0]
                            if dtuple[2] == inst_id and eq_node.split(':')[-1] == 'equality':
                                # find_equal_elements(drg_tuples, dtuple, )
                                for dtuple in drg_tuples:
                                    if dtuple[0] == eq_node:
                                        inst_id_left = dtuple[2]
                                        for dtuple in drg_tuples:
                                            if dtuple[2] == inst_id_left and dtuple[1] == 'instance':
                                                inst_argument = inst_id_left.split(':')[-1]
                                                instances.append(dtuple[0].split(':')[1] + '(' + inst_argument + ')')
                                                instances.append(inst_id + '=' + inst_id_left)
        # find surfaces k4 surface k4:e10 4 [ because ]
        if dtuple[2] == event_id and dtuple[1] == 'surface':
            surfaces.append(dtuple[5])
        # find referents and functions with a surface form: k32 function k32:e13 3 [ not ]
        # discourse_unit = ':'.join(event_id.split(':')[:-1])  # get the id of the discourse unit, e.g. k32
        # if dtuple[0] == discourse_unit and dtuple[2] == event_id and (dtuple[1] == 'function' or dtuple[1] == 'referent') and dtuple[5] != ']':
         #   connectives.append(dtuple[1] + ' "' + dtuple[5] + '"')
        current_triple = ['REL', 'INT', 'EXT']
    return them_roles_smart, them_roles, temporalities, relations, attributes,\
           instances, surfaces, propositions, connectives, connectives2

# todo function to generate readable triples


def get_propositions(drg_tuples, propositions, inst_id):
    # c97:theme:1 ext k9:p2
    # k9:p2 event c100:be:0 0 [ ]
    props = False
    count = 0  # count how many lines start with k9:p2
    for dtuple in drg_tuples:
        discourse_ref = dtuple[0]
        if discourse_ref == inst_id:
            count += 1
            line = dtuple
            if dtuple[1] == 'event':
                event_node = dtuple[2]
                for dtuple in drg_tuples:
                    if dtuple[0] == event_node:
                        propositions.append(dtuple[2])  # add event id
                        props = True
            elif dtuple[1] == 'dominates':  # case of the coordination -- can dominate two or more propositions
                triple = dtuple[1] + '(' + dtuple[0] + ', ' + dtuple[2] + ')'
                propositions.append(triple)
                propositions = get_propositions(drg_tuples, propositions, dtuple[2])
                props = True
            elif dtuple[1] == 'binary' or dtuple[1] == 'duplex':  # '['k37', 'binary', 'c148:imp', '0', '[', ']']
                triple = dtuple[1] + '(' + dtuple[0] + ', ' + dtuple[2] + ')'
                propositions.append(triple)
                bin_node = dtuple[2]
                for dtuple in drg_tuples:
                    if dtuple[0] == bin_node:
                        triple = dtuple[1] + '(' + dtuple[0] + ', ' + dtuple[2] + ')'
                        propositions.append(triple)
                        propositions = get_propositions(drg_tuples, propositions, dtuple[2])
                        props = True
            elif dtuple[1] == 'unary':
                triple = dtuple[1] + '(' + dtuple[0] + ', ' + dtuple[2] + ')'
                propositions.append(triple)
                un_node = dtuple[2]
                for dtuple in drg_tuples:
                    if dtuple[0] == un_node and dtuple[1] == 'scope':
                        triple = dtuple[1] + '(' + dtuple[0] + ', ' + dtuple[2] + ')'
                        propositions.append(triple)
                        propositions = get_propositions(drg_tuples, propositions, dtuple[2])
                        props = True

    if count == 1 and not props:
        if line[1] == 'referent':  # ['k3:p3', 'referent', 'k3:p3:p4', '1', '[', 'that', ']']
            triple = line[1] + '(' + line[0] + ', ' + line[2] + '), SF: ' + line[5]
            propositions.append(triple)
            propositions = get_propositions(drg_tuples, propositions, line[2])
            props = True
    if not props:
        propositions.append(inst_id + ': noEventsFound')
    return propositions


def get_temporalities(drg_tuples, temporalities, temp_type, ddtuple):
    '''Extract temporal relations from DRG'''
    # c27:temp_included:1 ext k1:t2 0 [ ]
    # c28:equality int k1:t2 0 [ ]
    # c28:equality ext k1:t1 0 [ ]
    # c26:now:1 arg k1:t1 0 [ ]
    if temp_type == 'temp_included' or temp_type == 'temp_overlap' or temp_type == 'temp_abut':
        for dtuple in drg_tuples:
            if dtuple[2] == ddtuple[2] and 'equality' in dtuple[0]:
                pred_type = dtuple[0].split(":")[-2]
                temporalities = find_equal_elements(drg_tuples, temporalities, dtuple)
            elif dtuple[2] == ddtuple[2] and ('temp_includes' in dtuple[0] or 'temp_before' in dtuple[0]):
               temp_type_upd = dtuple[0].split(':')[-2]
               temporalities = get_temporalities(drg_tuples, temporalities, temp_type_upd, dtuple)
    elif temp_type == 'temp_includes' or temp_type == 'temp_before':
        # c81:temp_before:1 ext k28:t6 0 [ ]
        # c81:temp_before:1 int k28:t1 1 [ ]
        for dtuple in drg_tuples:
            if dtuple == ddtuple:
                continue  # do not consider the element; we need to find its pair, but not itself
            if dtuple[0] == ddtuple[0]:
                if ddtuple[1] == 'int':
                    triple = dtuple[0].split(':')[-2] + '(' + ddtuple[2] + ', ' + dtuple[2] + ')'
                elif ddtuple[1] == 'ext':
                    triple = dtuple[0].split(':')[-2] + '(' + dtuple[2] + ', ' + ddtuple[2] + ')'
                else:
                    print('wrong format in temporal relations')
                temporalities.append(triple)
                inst_id = dtuple[2]
                for dtuple in drg_tuples:
                    if dtuple[2] == inst_id and 'equality' in dtuple[0]:
                        pred_type = dtuple[0].split(":")[-2]
                        temporalities = find_equal_elements(drg_tuples, temporalities, dtuple)
                    elif dtuple[2] == inst_id and (dtuple[1] == 'instance' or dtuple[1] == 'arg'):
                        inst_argument = dtuple[2].split(':')[-1]
                        temporalities.append(dtuple[0].split(':')[-2] + '(' + inst_argument + ')')
                break
    return temporalities


def find_equal_elements(drg_tuples, elements, ddtuple):
    # c28:equality int k1:t2 0 [ ]
    # c28:equality ext k1:t1 0 [ ]
    # c26:now:1 arg k1:t1 0 [ ]
    inst_id = ddtuple[2]
    eq_node = ddtuple[0]
    drg_tuples.remove(ddtuple)  # delete the element in order not to find it in the loop
    for dtuple in drg_tuples:
        if dtuple[0] == eq_node:
            inst_id_left = dtuple[2]
            elements.append(inst_id + '=' + inst_id_left)  # add equal elements
            for dtuple in drg_tuples:
                if dtuple[2] == inst_id_left and (dtuple[1] == 'instance' or dtuple[1] == 'arg'):
                    inst_argument = inst_id_left.split(':')[-1]
                    elements.append(dtuple[0].split(':')[1] + '(' + inst_argument + ')')
            break
    return elements


def get_sentences(curr_directory, event_arg, predicate, guess='no', recursion_depth=0):
    '''Read DRS xml file. extract offset of events and sentences'''
    tree = ETree.parse(curr_directory + 'en.drs.xml')
    root = tree.getroot()
    offset = ''
    for pred in root.iter('pred'):
        if pred.attrib['arg'] == event_arg and pred.attrib['symbol'] == predicate:
        # <pred arg="e2" symbol="land" type="v" sense="1"><indexlist><index pos="10">i1010</index></indexlist></pred>
        # <pred arg="e11" symbol="event" type="v" sense="0"><indexlist></indexlist></pred>
            if len(pred[0]) == 1:
                offset = pred[0][0].text
            elif len(pred[0]) == 0:
                offset = 'Not available'
            else:
                print("no offset found! for {} {} {}".format(curr_directory, event_arg, predicate))
    if not offset:
    # try to find offsets in unmatched DRSs by searching for the event with the id incremented by 1, 2, 3 or substracted by 1, 2, 3
        guess = 'yes'
        if recursion_depth == 0:
            new_event_arg = 'e' + str(int(event_arg[1:]) + 1)
            offset, guess = get_sentences(curr_directory, new_event_arg, predicate, 'yes', 1)
        elif recursion_depth == 1:
            new_event_arg = 'e' + str(int(event_arg[1:]) - 2)
            offset, guess = get_sentences(curr_directory, new_event_arg, predicate, 'yes', 2)
        elif recursion_depth == 2:
            new_event_arg = 'e' + str(int(event_arg[1:]) + 3)
            offset, guess = get_sentences(curr_directory, new_event_arg, predicate, 'yes', 3)
        elif recursion_depth == 3:
            new_event_arg = 'e' + str(int(event_arg[1:]) - 4)
            offset, guess = get_sentences(curr_directory, new_event_arg, predicate, 'yes', 4)
        elif recursion_depth == 4:
            new_event_arg = 'e' + str(int(event_arg[1:]) + 5)
            offset, guess = get_sentences(curr_directory, new_event_arg, predicate, 'yes', 5)
        elif recursion_depth == 5:
            new_event_arg = 'e' + str(int(event_arg[1:]) - 6)
            offset, guess = get_sentences(curr_directory, new_event_arg, predicate, 'yes', 6)
        elif recursion_depth == 6:
            new_event_arg = 'e' + str(int(event_arg[1:]) + 7)
            offset, guess = get_sentences(curr_directory, new_event_arg, predicate, 'yes', 7)
        else:
            offset = 'Not available'
            print("no offset found for {} {} {} due to xml-formatting".format(curr_directory, event_arg, predicate))

    return offset, guess


def calculate_stats(them_roles, roles_dict, ccg_category, ccg_cats):
    # Calculate stats of thematic roles
    # [agent-1(x25, e9), theme(e9, p4)]
    roles = []
    roles += [triple.split('(')[0] for triple in them_roles]
    roles_sorted = sorted(roles)
    roles_tuple = tuple(roles_sorted)
    if roles_tuple in roles_dict:
        roles_dict[roles_tuple] += 1
    else:
        roles_dict[roles_tuple] = 1
    # Calculate stats of ccg categories
    ccg_cats[ccg_category] += 1
    # Build abridged semantic representations: agent-1(e, x), theme(e, p)
    abr_roles = []
    if not them_roles:
        abr_roles += ['NoRolesFound']
    for triple in them_roles:
        role, two_args = triple.split('(')
        arg1, arg2 = two_args.split(', ')
        abr_roles += [role + '(' + arg1[0] + ', ' + arg2[0] + ')']
    abr_roles_tuple = tuple(sorted(abr_roles))
    return roles_dict, ccg_cats, abr_roles_tuple


def ccg_categories(token, offset, file_path):
    if token == 'EVENT':
        return 'NoOffset'
    # if token == 'EllipticalEvent':
    sent_num = int(offset[1:])//1000  # offset = i14007
    token_num = int(offset[-3:])
    ccg_cats = []
    sent = []
    with open(file_path+'en.tags', 'r') as f:
        for line in f:
            if line != '\n':
                sent.append(line.split('\t')[8])
            else:
                ccg_cats.append(sent)
                sent = []
    ccg_cats.append(sent)  # files don't end with the blank line
    ccg_cat = ccg_cats[sent_num - 1][token_num - 1]
    return ccg_cat


def normalise_ccg_cat(ccg_category):
    if '/' in ccg_category:
        cat_norm = ccg_category.split('/')[0]
        cat_norm = cat_norm.replace(')', '')
        cat_norm = cat_norm.replace('(', '')
    else:
        cat_norm = ccg_category
    if ccg_category == 'N/N':
        cat_norm = ccg_category
    return cat_norm


read_corpus(GMB_path)
print(semtypes)
print('Graph edges: {}'.format(edges))
print('Which in predicates: {}'.format(pred_which_count))
print('Which in words: {}'.format(word_which_count))
