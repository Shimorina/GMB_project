import os
import csv
import xml.etree.ElementTree as ET


GMB_path = '/home/anastasia/Documents/the_GMB_corpus/gmb-2.2.0/data_test/'


def read_corpus(GMB_path):
    count = 0
    with open('events.csv', 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Path', 'Token', 'Event', 'Offset', 'Event Predicate', 'Semantics', 'Basic Conditions'])
    for partition in os.listdir(GMB_path):
        partition_path = GMB_path + '/' + partition  # get path of each partition
        for entry in os.listdir(partition_path):
            met = partition_path + '/' + entry + '/en.met'
            drg = partition_path + '/' + entry + '/en.drg'
            tags = partition_path + '/' + entry + '/en.tags'
            drs = partition_path + '/' + entry + '/en.drs.xml'
            tokoff = partition_path + '/' + entry + '/en.tok.off'
            curr_directory = partition_path + '/' + entry + '/'
            drg_mining(drg, curr_directory)
            count += 1
    print('Number of docs in GMB: {}'.format(count))


def drg_mining(file_path, curr_directory):
    '''Read drg-file. Extract events with their relations.'''
    drg_tuples = []  # tuples of DRG
    with open(file_path, 'r') as f:
        next(f) # skip first three lines
        next(f)
        next(f)
        for line in f:
            if line != '\n':
                drg_tuples.append(line.rstrip().split())

    # Find events (by searching subtype event)
    events = []
    for tuple in drg_tuples:
        if tuple[1] == 'event':
            events.append(tuple[2])  # k3:p1 event c41:open:1 0

    for event in events:
        # Get a surface form of the event
        for tuple in drg_tuples:
            if tuple[0] == event:
                if tuple[1] == 'instance':
                    token = tuple[5]  # c52:spot:1 instance k3:p1:e5 3 [ spotting ]
                    event_id = tuple[2]
                    predicate = tuple[0].split(':')[1] + '(' + event_id.split(':')[-1] + ')'
                else:
                    print('wrong format')

        # Get event relations
        semantics, surface_forms = event_relation(drg_tuples, event_id)
        offset = get_sentences(curr_directory, event_id.split(':')[1], tuple[0].split(':')[1])
        with open('events.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            fpath_short = file_path.split('/')[-3] + '/' + file_path.split('/')[-2]
            csvwriter.writerow([fpath_short, token, event_id, offset, predicate, ', '.join(semantics), ', '.join(surface_forms)])


semtypes = set()
edges = set()


def event_relation(drg_tuples, event_id):
    '''This is a function, walking the DR graph to extract all the event relations with their attributes.'''
    current_triple = []
    relations = []
    surface_entities = []
    global semtypes
    # c54:patient:1 int k3:p1:e5 4 [ ]
    for tuple in drg_tuples:
        edges.add(tuple[1])
        if tuple[2] == event_id and tuple[0][0] != 'k' and tuple[1] != 'instance':  # tuple does not start with "k" and the edge is not of type "instance"
            current_triple.append(event_id.split(':')[1])  # e5
            try:
                current_triple.append(tuple[0].split(':')[-2])  # patient
            except:
                print(tuple)
            sem_id = tuple[0]
            # c54:patient:1 ext k3:p1:x16 0 [ ]
            for tuple in drg_tuples:
                if tuple[0] == sem_id and tuple[1] == ('ext' or 'int'):
                    current_triple.append(tuple[2].split(':')[-1])  # x16
                    inst_id = tuple[2]
                    triple = current_triple[1] + '(' + current_triple[0] + ', ' + current_triple[2] + ')'
                    relations.append(triple)  # triple is filled
                    semtypes.add(current_triple[1])

                    # c49:people:1 instance k3:p1:x16 2 [ people ]
                    for tuple in drg_tuples:
                        if tuple[2] == inst_id and tuple[1] == 'instance':
                            argument = inst_id.split(':')[-1]  # x16
                            surface_entities.append(tuple[0].split(':')[1] + '(' + argument + ')')  # get lemma "people"
        current_triple = []


    return relations, surface_entities

# function to generate readable triples


def get_sentences(curr_directory, event_arg, predicate):
    '''Read DRS xml file. extract offset of events and sentences'''
    tree = ET.parse(curr_directory + 'en.drs.xml')
    root = tree.getroot()
    for pred in root.iter('pred'):
        if pred.attrib['arg'] == event_arg and pred.attrib['symbol'] == predicate:
            offset = pred[0][0].text
            #  <pred arg="e2" symbol="land" type="v" sense="1"><indexlist><index pos="10">i1010</index></indexlist></pred>
        else:
            n = pred.attrib['arg']
            f = pred.attrib['symbol']
    return offset



read_corpus(GMB_path)
print(semtypes)
print(edges)