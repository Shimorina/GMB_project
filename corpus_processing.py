import os
import csv


GMB_path = '/home/anastasia/Documents/the_GMB_corpus/gmb-2.2.0/data_test/'


def read_corpus(GMB_path):
    count = 0
    with open('events.csv', 'w+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Path', 'Token', 'Event', 'Event Predicate', 'Semantics'])
    for partition in os.listdir(GMB_path):
        partition_path = GMB_path + '/' + partition  # get path of each partition
        for entry in os.listdir(partition_path):
            met = partition_path + '/' + entry + '/en.met'
            drg = partition_path + '/' + entry + '/en.drg'
            tags = partition_path + '/' + entry + '/en.tags'
            drg_mining(drg)
            count += 1
    print('Number of docs in GMB: {}'.format(count))


def drg_mining(file_path):
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
        semantics = event_relation(drg_tuples, event_id, node=event_id, relations=[], current_triple=[], triple_num=1)
        with open('events.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            fpath_short = file_path.split('/')[-3] + '/' + file_path.split('/')[-2]
            csvwriter.writerow([fpath_short, token, event_id, predicate] + semantics)


def event_relation(drg_tuples, event_id, node, relations, current_triple, triple_num):
    '''This is a recursive function, walking the DR graph to extract all the event relations with their attributes.'''
    if triple_num == 1:  # c54:patient:1 int k3:p1:e5 4 [ ]
        for tuple in drg_tuples:
            if tuple[2] == node and tuple[0][0] != 'k' and tuple[1] != 'instance':  # tuple does not start with "k" and the edge is not of type "instance"
                current_triple.append(node.split(':')[1])  # e5
                try:
                    current_triple.append(tuple[0].split(':')[-2])  # patient
                except:
                    print(tuple)
                node = tuple[0]
                #return event_relation(drg_tuples, event_id, node, relations, current_triple, 2)


    elif triple_num == 2:  # c54:patient:1 ext k3:p1:x16 0 [ ]
        for tuple in drg_tuples:
            if tuple[0] == node and tuple[1] == 'ext':
                current_triple.append(tuple[2].split(':')[-1])  # x16
                node = tuple[2]
                triple = current_triple[1] + '(' + current_triple[0] + ', ' + current_triple[2] + ')'
                relations.append(triple)  # triple is filled
                current_triple = []
                #return event_relation(drg_tuples, event_id, node, relations, current_triple, 3)
                break

    elif triple_num == 3:  # c49:people:1 instance k3:p1:x16 2 [ people ]
        for tuple in drg_tuples:
            if tuple[2] == node and tuple[1] == 'instance':
                argument = node.split(':')[-1]  # x16
                relations.append(tuple[0].split(':')[1] + '(' + argument + ')')  # get lemma "people"
                # return event_relation(drg_tuples, event_id, event_id, relations, [], 1)
                break

    return relations

# function to generate readable triples

read_corpus(GMB_path)