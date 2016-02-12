import os
import csv
import xml.etree.ElementTree as ETree


GMB_path = '/home/anastasia/Documents/the_GMB_corpus/gmb-2.2.0/data_test/'
roles = ['agent', 'asset', 'attribute', 'beneficiary', 'cause', 'co-agent', 'co-theme', 'destination', 'extent', 'experiencer', 'frequency', 'goal', 'initial_location', 'instrument', 'location', 'manner', 'material', 'patient', 'path', 'pivot', 'product', 'recipient', 'result', 'source', 'stimulus', 'time', 'topic', 'theme', 'trajectory', 'value', 'agent-1', 'asset-1', 'attribute-1', 'beneficiary-1', 'cause-1', 'co-agent-1', 'co-theme-1', 'destination-1', 'extent-1', 'experiencer-1', 'frequency-1', 'goal-1', 'initial_location-1', 'instrument-1', 'location-1', 'manner-1', 'material-1', 'patient-1', 'path-1', 'pivot-1', 'product-1', 'recipient-1', 'result-1', 'source-1', 'stimulus-1', 'time-1', 'topic-1', 'theme-1', 'trajectory-1', 'value-1']


def read_corpus(gmb_path):
    count = 0
    with open('events.csv', 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Path', 'Token', 'Event', 'Offset', 'Event Predicate', 'Thematic roles', 'Temporalities', 'Other Semantics', 'Attributes', 'Entities', 'Propositions', 'Surfaces', 'Sentence', 'Guess Offset'])
    for partition in os.listdir(gmb_path):
        partition_path = gmb_path + '/' + partition  # get path of each partition
        for entry in os.listdir(partition_path):
            curr_directory = partition_path + '/' + entry + '/'
            count += 1
            drg_mining(curr_directory)
    print('Number of docs in GMB: {}'.format(count))


def drg_mining(file_path):
    '''Read drg-file. Extract events with their relations.
    file_path '''

    # Read en.tags file and build a list of sentences
    sent = []
    sentence = []
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
        them_roles, temporalities, semantics, attributes, instances, surfaces, propositions = event_relation(drg_tuples, event_id)
        # Get offset of the event
        offset, guess = get_sentences(file_path, pure_event_id, predicate)  # i16014
        # Get sentence with the event in question
        if offset != 'Not available':
            target_sent = sent[int(offset[1:-3]) - 1]
        else:
            target_sent = 'Unknown'

        with open('events.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            fpath_short = file_path.split('/')[-3] + '/' + file_path.split('/')[-2]
            csvwriter.writerow([fpath_short, token, event_id, offset, predicate_arg, ', '.join(them_roles), ', '.join(temporalities), ', '.join(semantics), ', '.join(attributes), ', '.join(instances), ', '.join(propositions), ', '.join(surfaces), ' '.join(target_sent), guess])


semtypes = set()
edges = set()


def event_relation(drg_tuples, event_id):
    '''This is a function, walking the DR graph to extract all the event relations with their attributes.'''
    current_triple = ['REL', 'INT', 'EXT']  # contain conditions ['agent', 'e1', 'x4'] : e.g. "agent (e1, x4)", "temp_included(e20, t13)"
    relations = []
    them_roles = []
    temporalities = []
    instances = []
    attributes = []
    surfaces = []
    propositions = []
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
                    attributes.append(sem_id.split(':')[-2] + '(' + event_id.split(':')[-1] + ')')
                else:
                    attributes.append(sem_id.split(':')[-2] + '(EllipticalAttr)')
                continue
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
                        elif current_triple[0].startswith('temp'):
                            temporalities.append(triple)
                        else:
                            relations.append(triple)  # triple is filled
                        semtypes.add(current_triple[0])

                        for dtuple in drg_tuples:
                            if dtuple[2] == inst_id:
                                inst_argument = inst_id.split(':')[-1]  # x16
                                if dtuple[1] == 'instance': # c49:people:1 instance k3:p1:x16 2 [ people ]
                                    instances.append(dtuple[0].split(':')[1] + '(' + inst_argument + ')')  # get lemma "people"
                                elif dtuple[1] == 'arg':    # c19:nearly:1  arg k1:x7   2   [   nearly ]
                                    attributes.append(dtuple[0].split(':')[1] + '(' + inst_argument + ')')
                                elif dtuple[1] == 'surface':  # k8  surface k8:e17  2   [   and   ]
                                    surfaces.append(dtuple[5])

                    elif argument == 'ext' and dtuple[1] == 'int':
                        current_triple[1] = dtuple[2].split(':')[-1]  # x16
                        inst_id = dtuple[2]
                        triple = current_triple[0] + '(' + current_triple[1] + ', ' + current_triple[2] + ')'
                        if current_triple[0] in roles:
                            them_roles.append(triple)
                        elif current_triple[0].startswith('temp'):
                            temporalities.append(triple)
                        else:
                            relations.append(triple)  # triple is filled
                        semtypes.add(current_triple[0])

                        # c209:equality ext k76:x37
                        for dtuple in drg_tuples:
                            eq_node = dtuple[0]
                            if dtuple[2] == inst_id and eq_node.split(':')[-1] == 'equality':
                                for dtuple in drg_tuples:
                                    if dtuple[0] == eq_node:
                                        inst_id_left = dtuple[2]
                                        for dtuple in drg_tuples:
                                            if dtuple[2] == inst_id_left and dtuple[1] == 'instance':
                                                inst_argument = inst_id_left.split(':')[-1]
                                                instances.append(dtuple[0].split(':')[1] + '(' + inst_argument + ')')
                                                instances.append(inst_id + '=' + inst_id_left)

        current_triple = ['REL', 'INT', 'EXT']
    return them_roles, temporalities, relations, attributes, instances, surfaces, propositions

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
    if count == 1 and not props:
        if line[1] == 'unary':
            triple = line[1] + '(' + line[0] + ', ' + line[2] + ')'
            propositions.append(triple)
            for dtuple in drg_tuples:
                if dtuple[0] == line[2] and dtuple[1] == 'scope':
                    triple = dtuple[1] + '(' + dtuple[0] + ', ' + dtuple[2] + ')'
                    propositions.append(triple)
                    propositions = get_propositions(drg_tuples, propositions, dtuple[2])
                    props = True
        elif line[1] == 'dominates':
            triple = line[1] + '(' + line[0] + ', ' + line[2] + ')'
            propositions.append(triple)
            propositions = get_propositions(drg_tuples, propositions, line[2])
            props = True
        elif line[1] == 'referent':  # ['k3:p3', 'referent', 'k3:p3:p4', '1', '[', 'that', ']']
            triple = line[1] + '(' + line[0] + ', ' + line[2] + '), SF: ' + line[5]
            propositions.append(triple)
            propositions = get_propositions(drg_tuples, propositions, line[2])
            props = True
        elif line[1] == 'binary' or 'duplex':  # '['k37', 'binary', 'c148:imp', '0', '[', ']']
            triple = line[1] + '(' + line[0] + ', ' + line[2] + ')'
            propositions.append(triple)
            for dtuple in drg_tuples:
                if dtuple[0] == line[2]:
                    triple = dtuple[1] + '(' + dtuple[0] + ', ' + dtuple[2] + ')'
                    propositions.append(triple)
                    propositions = get_propositions(drg_tuples, propositions, dtuple[2])
                    props = True
        else:
            print(line)
    if not props:
        propositions.append('noEventsFound')
    return propositions


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


read_corpus(GMB_path)
print(semtypes)
print("Graph edges: {}".format(edges))
