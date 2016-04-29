import os
import csv
import string
from natsort import natsorted
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
                            'Other Semantics [surface form if any]', 'Other Semantics (Two events)',
                            'Attributes ("arg" edges) [surface form if any]', 'Entities',
                            'Propositions', 'Surfaces', 'Function words', 'Referents',
                            'Sentence', 'Guess Offset', 'Pronominalisation', 'Temporalities'])
    with open('ccg_categories_all.csv', 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Path', 'Token', 'Offset', 'CCG cat', 'Normalised cat', 'Refined cat',
                            'Training cat', 'Agent-1', 'Patient-1', 'Sentence', 'Guess Offset'])
    with open('training_data_pairs_all.txt', 'w+') as f:
        f.write('#Agent\tPatient\tAgent-1\tPatient-1\tTheme\tTheme-1\tRecipient\tRecipient-1\tTopic\tSyntacticLabel\n')
    with open('training_data_sequences_all.txt', 'w+') as f:
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
    lemmas_sent = []
    lemmas_file = []
    tags_sent = []
    tags_file = []
    file_train_set = {}
    global roles_dict
    global ccg_cats
    global sems_synt
    global word_which_count
    global pred_which_count
    with open(file_path+'en.tags', 'r') as f:
        for line in f:
            if line != '\n':
                token, tag, lemma, *rest = line.split('\t')
                sentence += [token]
                lemmas_sent += [lemma]
                tags_sent += [tag]
            else:
                sent += [sentence]
                sentence = []
                lemmas_file += [lemmas_sent]
                lemmas_sent = []
                tags_file += [tags_sent]
                tags_sent = []
        sent += [sentence]  # file does not end with an empty line
        lemmas_file += [lemmas_sent]
        tags_file += [tags_sent]
    ccg_categories_file = ccg_categories(file_path)
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
        them_roles_smart, them_roles, temporalities, semantics, semantics_events, attributes,\
        instances, surfaces, propositions, connectives, connectives2, pronoms = event_relation(drg_tuples, event_id)
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
            csvwriter.writerow([fpath_short, token, event_id, offset, predicate_arg, ', '.join(them_roles), ', '.join(semantics), ', '.join(semantics_events), ', '.join(attributes), ', '.join(instances), ', '.join(propositions), '||'.join(surfaces), ' '.join(connectives), ' '.join(connectives2), ' '.join(target_sent), guess, ', '.join(pronoms), ', '.join(temporalities)])

        # Get the ccg category for the event
        ccg_category, ccg_cat_norm, refined_cat, train_cat, wh_pretend, wh_obj_pret = profiling_ccg_category(token, offset, ccg_categories_file, lemmas_file, tags_file, sent, them_roles_smart, pronoms)
        with open('ccg_categories_all.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([fpath_short, token, offset, ccg_category, ccg_cat_norm, refined_cat, train_cat, wh_pretend, wh_obj_pret, ' '.join(target_sent), guess])
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
        # put all events of one file to dict with the structure: i450 : [[agent:x1, patient:x56, recipient:x4], NP/S]
        # we do not consider events marked as EVENT
        if offset != 'Not available' and train_cat != 'Not available':
            file_train_set[offset] = [them_roles_smart, train_cat]

    # write training data to file
    crf_data_pairs(file_train_set, fpath_short)
    # write training data for CRFs to file
    crf_data(file_train_set, fpath_short)


semtypes = set()
edges = set()


def event_relation(drg_tuples, event_id):
    """
    Walk the DR graph and extract all event relations with their attributes.
    :param drg_tuples: the DRG -- a list of tuples where two nodes are connected with an edge
    :param event_id: id of the event in a DRG tuple, e.g. k3:p1:e5
    :return:
    """
    current_triple = ['REL', 'INT', 'EXT']  # conditions ['agent', 'e1', 'x4'] : e.g. "agent (e1, x4)", "temp_included(e20, t13)"
    them_roles = []
    them_roles_smart = []  # role:event_arg, e.g. agent-1:e9
    temporalities = []
    relations = []
    relations_events = []
    attributes = []
    instances = []
    surfaces = []
    propositions = []
    connectives = []
    connectives2 = []
    pronoms = []
    global semtypes
    global roles
    # c54:patient:1 int k3:p1:e5 4 [ ]
    for dtuple in drg_tuples:
        edges.add(dtuple[1])
        # Exclude referent and condition tuples ("k6 referent k6:e1") and argument tuples of type "instance"
        # tuple should start with "c" and the edge is not of type "instance"
        if dtuple[2] == event_id and dtuple[0].startswith('c') and dtuple[1] != 'instance':
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
                        th_role = current_triple[0]
                        triple = current_triple[0] + '(' + current_triple[1] + ', ' + current_triple[2] + ')'
                        if current_triple[0] in roles:
                            them_roles.append(triple)
                            them_roles_smart.append(current_triple[0] + ':' + current_triple[2])
                        elif current_triple[0].startswith('temp'):
                            temporalities.append(triple)
                            temporalities = get_temporalities(drg_tuples, temporalities, current_triple[0], dtuple)
                        # add other semantics
                        else:
                            # add surface form if exists
                            if dtuple[5] != ']':
                                triple += ' [' + dtuple[5] + ']'
                            if current_triple[1].startswith('e') and not current_triple[2].startswith('x'):
                                relations_events.append(triple)
                            elif not current_triple[1].startswith('x') and current_triple[2].startswith('e'):
                                relations_events.append(triple)
                            else:
                                relations.append(triple)  # triple is filled
                        semtypes.add(current_triple[0])

                        for dtuple in drg_tuples:
                            if dtuple[2] == inst_id:
                                inst_argument = inst_id.split(':')[-1]  # x16
                                if dtuple[1] == 'instance':  # c49:people:1 instance k3:p1:x16 2 [ people ]
                                    arg_lemma = dtuple[0].split(':')[1]  # get lemma "people"
                                    instance_with_argument = arg_lemma + '(' + inst_argument + ')'
                                    # avoid two identical instances in one DRS
                                    if instance_with_argument not in instances:
                                        instances.append(instance_with_argument)
                                        pronom = pronominalisation_check(arg_lemma)
                                        pronoms.append(th_role + '|||' + pronom)
                                elif dtuple[1] == 'arg':    # c19:nearly:1  arg k1:x7   2   [   nearly ]
                                    attributes.append(dtuple[0].split(':')[1] + '(' + inst_argument + ')')
                                elif dtuple[1] == 'surface':  # k8  surface k8:e17  2   [   and   ]
                                    surfaces.append(dtuple[5])

                                if dtuple[1] == 'referent' and dtuple[5] != ']':  # k2 referent k2:x11 1 [ which ]
                                    connectives.append('referent "' + dtuple[5] + '" (' + inst_argument + ')')

                    elif argument == 'ext' and dtuple[1] == 'int':
                        current_triple[1] = dtuple[2].split(':')[-1]  # x16
                        inst_id = dtuple[2]
                        th_role = current_triple[0]
                        triple = current_triple[0] + '(' + current_triple[1] + ', ' + current_triple[2] + ')'
                        if current_triple[0] in roles:
                            them_roles.append(triple)
                            them_roles_smart.append(current_triple[0] + ':' + current_triple[1])
                        elif current_triple[0].startswith('temp'):
                            temporalities.append(triple)
                            temporalities = get_temporalities(drg_tuples, temporalities, current_triple[0], dtuple)
                        # add other semantics
                        else:
                            if current_triple[1].startswith('e') and not current_triple[2].startswith('x'):
                                relations_events.append(triple)
                            elif not current_triple[1].startswith('x') and current_triple[2].startswith('e'):
                                relations_events.append(triple)
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
                                                arg_lemma = dtuple[0].split(':')[1]  # get lemma "people"
                                                instance_with_argument = arg_lemma + '(' + inst_argument + ')'
                                                # avoid two identical instances in one DRS
                                                if instance_with_argument not in instances:
                                                    instances.append(instance_with_argument)
                                                    pronom = pronominalisation_check(arg_lemma)
                                                    pronoms.append(th_role + '|||' + pronom)
                                                equal_inst = inst_id + '=' + inst_id_left
                                                if equal_inst not in instances:
                                                    instances.append(equal_inst)

        # find surfaces k4 surface k4:e10 4 [ because ]
        if dtuple[2] == event_id and dtuple[1] == 'surface':
            surfaces.append(dtuple[5])
        # find referents and functions with a surface form: k32 function k32:e13 3 [ not ]
        # discourse_unit = ':'.join(event_id.split(':')[:-1])  # get the id of the discourse unit, e.g. k32
        # if dtuple[0] == discourse_unit and dtuple[2] == event_id and (dtuple[1] == 'function' or dtuple[1] == 'referent') and dtuple[5] != ']':
        #  connectives.append(dtuple[1] + ' "' + dtuple[5] + '"')
        current_triple = ['REL', 'INT', 'EXT']
    return them_roles_smart, them_roles, temporalities, relations, relations_events, attributes,\
           instances, surfaces, propositions, connectives, connectives2, pronoms

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
        # <pred arg="e2" symbol="land" type="v" sense="1"><indexlist><index pos="10">i1010</index></indexlist></pred>
        # <pred arg="e11" symbol="event" type="v" sense="0"><indexlist></indexlist></pred>
        if pred.attrib['arg'] == event_arg and pred.attrib['symbol'] == predicate:
            if len(pred[0]) == 1:
                offset = pred[0][0].text
            elif len(pred[0]) == 0:
                offset = 'Not available'
            else:
                print("no offset found! for {} {} {}".format(curr_directory, event_arg, predicate))
    if not offset:
        # try to find offsets in unmatched DRSs by searching for the event
        # with the id incremented by 1, 2, 3 or substracted by 1, 2, 3
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


def pronominalisation_check(lemma):
    pronouns = ['male', 'female', 'thing']   # he, she, it
    if lemma in pronouns:
        return lemma
    else:
        return 'False'


def ccg_categories(file_path):
    """
    Read the file en.tags and extract CCG categories
    :param file_path:
    :return: list of sentences, each of them is a list of CCG categories
    """
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
    return ccg_cats


def profiling_ccg_category(token, offset, ccg_categories_file, lemmas, ptb_tags, sentences_file, them_roles_args, pronouns):
    """ Do the profiling of existing CCG categories.
    :param token: the surface form of the event; e.g. given, mounts, made
    :param offset: the event offset; e.g. i15002
    :param ccg_categories_file: list of sentences where each of them is a list of ccg cats for all tokens
    :param lemmas: list of sentences in a file; each sentence is a list of lemmas
    :param sentences_file: list of sentences in a file; each sentence is a list of tokens
    :param them_roles_args: list of thematic roles for the event, e.g. [agent-1:x4, patient:x7]
    :return: ccg category of a current token and suggested ccg category after the profiling
    """
    # don't consider events...
    if token == 'EVENT':
        return 'NoOffset', 'Not available', 'Not available', 'Not available', 'NA', 'NA'
    # if token == 'EllipticalEvent':
    refined_cat_wh = False  # cases of "John who eats an apple"
    wh_subj_flag = False
    wh_obj_flag = False
    pro_subj_flag = False
    pro_obj_flag = False
    pro_subj_recip_flag = False
    pro_obj_recip_flag = False
    sent_num = int(offset[1:])//1000  # offset = i14007
    token_num = int(offset[-3:])
    tk_0position = token_num - 1
    tk_1position = token_num - 2
    tk_2position = token_num - 3
    tk_3position = token_num - 4
    tk_4position = token_num - 5
    ccg_cats_sent = ccg_categories_file[sent_num - 1]
    # CCG tag of the current token
    ccg_cat = ccg_cats_sent[tk_0position]
    ccg_norm = normalise_ccg_cat(ccg_cat)  # we don't need what is on the right part
    sentence = lemmas[sent_num - 1]  # get list of lemmas for the sentence considered
    tags = ptb_tags[sent_num - 1]  # list of tags for the sentence considered
    tokens = sentences_file[sent_num - 1]  # list of tokens for the sentence considered
    lemma_1 = sentence[tk_1position]
    lemma_1_cat = ccg_cats_sent[tk_1position]
    lemma_1_tag = tags[tk_1position]
    token_1 = tokens[tk_1position]

    lemma_2_tag = tags[tk_2position]
    lemma_2 = sentence[tk_2position]
    lemma_2_cat = ccg_cats_sent[tk_2position]
    token_2 = tokens[tk_2position]

    lemma_3 = sentence[tk_3position]
    lemma_3_cat = ccg_cats_sent[tk_3position]
    lemma_3_tag = tags[tk_3position]
    token_3 = tokens[tk_3position]
    # profiling for five categories: S[dcl]\NP, S[b]\NP, S[pss]\NP, S[ng]\NP, S[pt]\NP
    if ccg_norm == 'S[dcl]\\NP':
        refined_cat = ccg_norm
        refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_0position)
    elif ccg_norm == 'S[b]\\NP':
        prev_token_cat_norm = normalise_ccg_cat(lemma_1_cat)
        # search for modals: could speak, will return
        if lemma_1_tag == 'MD' and '/(S[b]\\NP)' in lemma_1_cat:
            prev_token_cat_norm = normalise_ccg_cat(lemma_1_cat)
            if prev_token_cat_norm == 'S[dcl]\\NP':
                refined_cat = 'S[dcl-modal]\\NP'
            else:
                refined_cat = prev_token_cat_norm + '\t[modified-bare]'
            refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_1position)
        # negation or adverbs: did not elaborate, does not play, would likely spread
        elif lemma_1_tag == 'RB' and '/(S[b]\\NP)' in lemma_2_cat:
            prev_token_cat_norm = normalise_ccg_cat(lemma_2_cat)
            if prev_token_cat_norm == 'S[dcl]\\NP':
                refined_cat = 'S[dcl-neg/modal]\\NP'
            elif prev_token_cat_norm == 'S[to]\\NP':
                refined_cat = 'S[bare-to-Inf-2token]\\NP'
            else:
                refined_cat = prev_token_cat_norm + '\t[modified-bare-2]'
            refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_2position)
        # begin to drive, but exclude "has to set"
        elif prev_token_cat_norm == 'S[to]\\NP' and lemma_2 != 'have':
            refined_cat = 'S[bare-to-Inf]\\NP'
        # have to + Inf
        elif prev_token_cat_norm == 'S[to]\\NP' and lemma_2 == 'have':
            refined_cat = 'S[dcl-have-to-Inf]\\NP'
            refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_2position)
        # coordination: would disarm and return, will probably not move
        elif lemma_3_tag == 'MD' and '/(S[b]\\NP)' in lemma_3_cat and (lemma_1_cat == 'conj' or lemma_2_tag == 'RB'):
            refined_cat = 'S[dcl-modal-coord]\\NP'
            refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_3position)
        else:
            refined_cat = ccg_norm + '\t[not changed]'

    elif ccg_norm == 'S[pss]\\NP':
        # search for passive: 'to be' in previous or previous but one token
        if lemma_1 == 'be' and '/(S[pss]\\NP)' in lemma_1_cat:
            refined_cat = 'S[pss-dcl]\\NP'
            refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_1position)
            # case of "which had been trained"
            if not refined_cat_wh:
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_2position)
        # is not hurt, were later rescued, but exclude "were left stranded, have been reported killed"
        elif lemma_2 == 'be' and '/(S[pss]\\NP)' in lemma_2_cat and lemma_1_tag == 'RB':
            refined_cat = 'S[pss-dcl-2token]\\NP'
            refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_2position)
        # was spotted and prevented; was also brutally attacked; were no longer bound, but exclude "be held as scheduled"
        elif lemma_3 == 'be' and '/(S[pss]\\NP)' in lemma_3_cat and (lemma_1_cat == 'conj' or lemma_2_tag == 'RB'):
            refined_cat = 'S[pss-dcl-3token]\\NP'
            refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_3position)
        else:
            refined_cat = ccg_norm + '\t[not changed]'

    elif ccg_norm == 'S[ng]\\NP':
        # search for continuous tenses
        # is making
        if lemma_1 == 'be' and '/(S[ng]\\NP)' in lemma_1_cat:
            prev_token_cat_norm = normalise_ccg_cat(lemma_1_cat)
            if prev_token_cat_norm == 'S[dcl]\\NP':
                refined_cat = 'S[dcl-continuous]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_1position)
            # have been fighting, has since been struggling
            elif token_1 == 'been':
                refined_cat = 'S[dcl-perfect_continuous-1token]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_2position)
            #  could be developing
            #  todo go to the case of pt or b
            else:
                refined_cat = prev_token_cat_norm + '\t[modified]'
        # is not making, is almost going, but exclude "is considering boosting"
        elif lemma_2 == 'be' and '/(S[ng]\\NP)' in lemma_2_cat and lemma_1_tag == 'RB':
            prev_token_cat_norm = normalise_ccg_cat(lemma_2_cat)
            if prev_token_cat_norm == 'S[dcl]\\NP':
                refined_cat = 'S[dcl-continuous-2token]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_2position)
            # the perfect continuous tense: had [not] been narrowly observing
            elif token_2 == 'been':
                refined_cat = 'S[dcl-perfect_continuous-2token]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_2position)
            else:
                refined_cat = prev_token_cat_norm + '\t[modified-2]'
        # coordination: are regressing or lagging
        elif lemma_3 == 'be' and '/(S[ng]\\NP)' in lemma_3_cat and (lemma_1_cat == 'conj' or lemma_2_tag == 'RB'):
            prev_token_cat_norm = normalise_ccg_cat(lemma_3_cat)
            if prev_token_cat_norm == 'S[dcl]\\NP':
                refined_cat = 'S[dcl-continuous-3token]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_3position)
            # the perfect continuous tense: had been holding and interrogating
            elif token_3 == 'been':
                refined_cat = 'S[dcl-perfect_continuous-3token]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_4position)
            else:
                refined_cat = prev_token_cat_norm + '\t[modified-3]'
        #negation with adv:  are not just paying, is not completely withdrawing
        elif lemma_3 == 'be' and '/(S[ng]\\NP)' in lemma_3_cat and lemma_2 == 'not':
            prev_token_cat_norm = normalise_ccg_cat(lemma_3_cat)
            if prev_token_cat_norm == 'S[dcl]\\NP':
                refined_cat = 'S[dcl-continuous-3token--neg]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_3position)
            else:
                refined_cat = prev_token_cat_norm + '\t[modified-3]'
        else:
            refined_cat = ccg_norm + '\t[not changed]'

    elif ccg_norm == 'S[pt]\\NP':
        # search for perfectives
        if lemma_1 == 'have' and '/(S[pt]\\NP)' in lemma_1_cat:
            prev_token_cat_norm = normalise_ccg_cat(lemma_1_cat)
            if prev_token_cat_norm == 'S[dcl]\\NP':
                refined_cat = 'S[dcl-perfect]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_1position)
            # is known to have died, should [not] have lost,
            else:
                refined_cat = prev_token_cat_norm + '\t[modified]'
        # has since cooperated
        elif lemma_2 == 'have' and '/(S[pt]\\NP)' in lemma_2_cat:
            prev_token_cat_norm = normalise_ccg_cat(lemma_2_cat)
            if prev_token_cat_norm == 'S[dcl]\\NP':
                refined_cat = 'S[dcl-perfect-2token]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_2position)
            else:
                refined_cat = prev_token_cat_norm + '\t[modified-2]'  # having just surfeited, appears to have largely boycotted, may have accidentally strayed
        # has almost completely eliminated; have burned and dragged
        elif lemma_3 == 'have' and '/(S[pt]\\NP)' in lemma_3_cat:
            prev_token_cat_norm = normalise_ccg_cat(lemma_3_cat)
            if prev_token_cat_norm == 'S[dcl]\\NP':
                refined_cat = 'S[dcl-perfect-3token]\\NP'
                refined_cat_wh = wh_check(ccg_cats_sent, tags, curr_position=tk_3position)
            else:
                refined_cat = prev_token_cat_norm + '\t[modified-3]'
        else:
            refined_cat = ccg_norm + '\t[not changed]'
    # we don't do the profiling for infrequent categories
    else:
        refined_cat = ccg_norm
    # reduce info in the refined category for training data
    train_cats_dcl = ['S[dcl-perfect-3token]\\NP', 'S[dcl-perfect-2token]\\NP', 'S[dcl-perfect]\\NP',
                      'S[dcl-continuous-3token--neg]\\NP', 'S[dcl-perfect_continuous-3token]\\NP',
                      'S[dcl-continuous-3token]\\NP', 'S[dcl-perfect_continuous-2token]\\NP',
                      'S[dcl-continuous-2token]\\NP', 'S[dcl-perfect_continuous-1token]\\NP', 'S[dcl-continuous]\\NP',
                      'S[dcl-have-to-Inf]\\NP', 'S[dcl-neg/modal]\\NP', 'S[dcl-modal]\\NP', 'S[dcl-modal-coord]\\NP',]
    train_cats_bare = ['S[bare-to-Inf]\\NP', 'S[bare-to-Inf-2token]\\NP']
    train_cats_pss = ['S[pss-dcl-3token]\\NP', 'S[pss-dcl-2token]\\NP', 'S[pss-dcl]\\NP']
    # delete comments like [not changed], [modified]
    if '\t' in refined_cat:
        training_cat = refined_cat.split('\t')[0]
    else:
        training_cat = refined_cat
    if training_cat in train_cats_dcl:
        training_cat = 'S[dcl]\\NP'
    elif training_cat in train_cats_bare:
        training_cat = 'S[b]\\NP'
    elif training_cat in train_cats_pss:
        training_cat = 'S[pss-dcl]\\NP'
    # split declaratives into categories
    # if whSubj found, set its flag to true
    if refined_cat_wh:
        if training_cat == 'S[dcl]\\NP' or training_cat == 'S[pss-dcl]\\NP':
            wh_subj_flag = True
        else:
            print(training_cat)
    # add columns with agent-1 and patient-1
    wh_pretend = '---'
    wh_obj_pretend = '---'
    them_roles = []
    them_roles += [role.split(':')[0] for role in them_roles_args]
    for role in them_roles:
        if role == 'agent-1':
            if training_cat == 'S[dcl]\\NP':
                wh_pretend = 'whSubjCandidate'
            else:
                wh_pretend = 'agent-1'
        elif role == 'patient-1':
            # events which have patient-1 are dcl-whObj, e.g. the cash he got from the bank
            if training_cat == 'S[dcl]\\NP':
                wh_obj_flag = True
            else:
                wh_obj_pretend = 'patient-1'
    # generalise pronominalisation: replace M/F/thing with Pro
    for num, element in enumerate(pronouns):
        if element.endswith('male') or element.endswith('female') or element.endswith('thing'):
            pronouns[num] = element.split('|||')[0] + '|||Pro'
        else:
            pronouns[num] = element
    # modify CCG training category for declarative sentences if they are pronominalised
    if 'agent|||Pro' in pronouns and 'agent|||False' not in pronouns:
        if training_cat == 'S[dcl]\\NP':
            pro_subj_flag = True
        elif training_cat == 'S[pss-dcl]\\NP':
            pro_obj_flag = True
    if 'patient|||Pro' in pronouns and 'patient|||False' not in pronouns:
        if training_cat == 'S[dcl]\\NP':
            pro_obj_flag = True
        elif training_cat == 'S[pss-dcl]\\NP':
            pro_subj_flag = True
    # e.g. he was obliged to keep shares
    if training_cat == 'S[pss-dcl]\\NP' and 'theme' in them_roles and 'recipient' in them_roles and 'theme|||Pro' in pronouns:
        pro_subj_recip_flag = True
    # if there are agent and recipient as roles and no patient and recipient is pronominalised, then recipient is treated as ProObj
    # he urged them not to increase tensions -- them is a recipient
    if 'agent' in them_roles and 'recipient' in them_roles and 'patient' not in them_roles and 'recipient|||Pro' in pronouns:
        if training_cat in 'S[dcl]\\NP':
            pro_obj_recip_flag = True
    # add flags to categories
    if wh_subj_flag:
        training_cat = training_cat.replace('dcl', 'dcl-whSubj')
    if wh_obj_flag:
        training_cat = training_cat.replace('dcl', 'dcl-whObj')
    if pro_subj_flag:
        training_cat = training_cat.replace('dcl', 'dcl-ProSubj')
    if pro_obj_flag:
        training_cat = training_cat.replace('dcl', 'dcl-ProObj')
    if pro_subj_recip_flag:
        training_cat = training_cat.replace('dcl', 'dcl-ProSubjRecip')
    if pro_obj_recip_flag:
        training_cat = training_cat.replace('dcl', 'dcl-ProObjRecip')
    return ccg_cat, ccg_norm, refined_cat, training_cat, wh_pretend, wh_obj_pretend


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


def wh_check(ccg_tags, tags, curr_position):
    ccg_tag1 = ccg_tags[curr_position - 1]
    ccg_tag2 = ccg_tags[curr_position - 2]
    ccg_tag3 = ccg_tags[curr_position - 3]
    tag1 = tags[curr_position - 1]
    tag2 = tags[curr_position - 2]
    if ccg_tag1 == '(NP\\NP)/(S[dcl]\\NP)':
        refined_cat_wh = True
    # who once again are seeing; exclude "men who say they were kidnapped"
    elif ccg_tag2 == '(NP\\NP)/(S[dcl]\\NP)' and tag1 == 'RB':
        refined_cat_wh = True
    elif ccg_tag3 == '(NP\\NP)/(S[dcl]\\NP)' and tag1 == 'RB' and tag2 == 'RB':
        refined_cat_wh = True
    else:
        refined_cat_wh = False  # missed: policies which the group says include, attack that killed eight people and injured
    return refined_cat_wh


def crf_data_pairs(file_train_set, fpath_short):
    with open('training_data_pairs_all.txt', 'a') as f:
        placeholder_dict = ['agent', 'patient', 'agent-1', 'patient-1', 'theme',
                            'theme-1', 'recipient', 'recipient-1', 'topic']
        placeholder = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
        replacements = ['X', 'Y', 'Z', 'W', 'U']
        out = ''
        events = natsorted(file_train_set.keys())
        # for each pair of events, we will extract features and set them to 1 or to X
        # if roles of two events have the argument in common
        for i in range(len(events) - 1):
            event = events[i]
            next_event = events[i + 1]
            # events must be in one sentence! (i12003 -- 12 is equal to the sent number)
            if event[1:-3] != next_event[1:-3]:
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
        f.write(out)


def crf_data(file_train_set, fpath_short):
    with open('training_data_sequences_all.txt', 'a') as f:  # i450 : [[agent:x1, patient:x56, recipient:x4], NP/S]
        placeholder_dict = ['agent', 'patient', 'agent-1', 'patient-1', 'theme',
                            'theme-1', 'recipient', 'recipient-1', 'topic']
        placeholder = ['-', '-', '-', '-', '-', '-', '-', '-', '-', 'label']
        replacements = sorted(list(string.ascii_uppercase), reverse=True)  # the english alphabet
        out2 = ''
        # group events by sentences
        events = natsorted(file_train_set.keys())
        # list of empty dicts; take the last event to get the number of sentences
        sents = [{} for _ in range(int(events[-1][1:-3]))]
        # create dicts of events for each sentence: [{i100:[[roles], label], i109:[[roles], label], ...}, {i203:...}, {}, ...]
        for event in events:
            sent_number = int(event[1:-3])  # i13010, 13 stands for the sent number
            sents[sent_number - 1][event] = file_train_set[event]
        # for each sentence, extract features and set them to 1 or to X, Y, etc if roles have the argument in common
        for senten in sents:
            out2 += '#' + '--->'.join(natsorted(senten.keys())) + ' from ' + fpath_short + '\n'
            sent_args = []  # contains lists of args for each event
            # generate a placeholder for each event in a sentence
            sent_placeholder = [['-', '-', '-', '-', '-', '-', '-', '-', '-', 'label'] for i in range(len(senten))]
            for event in natsorted(senten):
                features, label = senten[event]
                arguments = [feat.split(':')[1] for feat in features]
                sent_args.append(arguments)
            # for each pair of events, check if they have elements in common
            common_elements = []
            for ev1 in sent_args:
                for ev2 in sent_args:
                    if ev1 != ev2:
                        common_elements += [item for item in set(ev1).intersection(ev2)]
            common_elements_unique = list(set(common_elements))
            # sort common args as a list of thematic roles in the placeholder_dict ??

            # replace common args with Z, Y, X, etc in a placeholder
            if common_elements_unique:
                j = 0  # iterate over replacements: Z, Y, X, etc
                for element in common_elements_unique:
                    # find the role of the common element and replace it with X
                    # do it for each event in a sentence
                    # i100 : [[agent:x1, patient:x5], label]
                    ev_counter = 0
                    for event in natsorted(senten):
                        features, label = senten[event]
                        for feat in features:
                            role, argument = feat.split(':')
                            if element == argument:
                                role_index = placeholder_dict.index(role)
                                sent_placeholder[ev_counter][role_index] = replacements[j]      
                        ev_counter += 1
                    j += 1
            # replace all other args of events with 1 except for Z, Y, X, etc
            ev_counter = 0
            for event in natsorted(senten):
                features, label = senten[event]
                # print(sent_placeholder)
                for feature in features:
                    role, argument = feature.split(':')
                    role_index = placeholder_dict.index(role)
                    if sent_placeholder[ev_counter][role_index] not in replacements:
                        sent_placeholder[ev_counter][role_index] = '1'
                # set label -- the last element in the sequence
                # print(sent_placeholder[ev_counter])
                sent_placeholder[ev_counter][9] = label
                ev_counter += 1
                # print(sent_placeholder)
            # write to output the sequence of events of a sentence
            for seq in sent_placeholder:
                out2 += '\t'.join(seq) + '\n'
            out2 += '\n'
        f.write(out2)
        # print(sent_placeholder)


read_corpus(GMB_path)
print(semtypes)
print('Graph edges: {}'.format(edges))
print('Which in predicates: {}'.format(pred_which_count))
print('Which in words: {}'.format(word_which_count))
