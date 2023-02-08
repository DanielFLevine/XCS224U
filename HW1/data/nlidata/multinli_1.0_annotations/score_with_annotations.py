from collections import defaultdict
import json
import sys
'''
    A quick and dirty script to score model output both for overall accuracy and for accuracy on each tag from
    our reference annotations.

    Inputs:
        annotations file: This should be one of the two *annotations.txt files included with this script.
        submission file: This should be your models output in csv format, with each line consisting of a "pairID,predicted_label" pair.
        data file: This should be the jsonl version of a MultiNLI dev set file. It is used to extract the correct labels.
'''

if len(sys.argv) < 4:
    print "Usage: score_with_annotations.py matched_annotations.txt PATH_TO_matched_submission.csv PATH_TO_matched.jsonl"

annotations_filename = sys.argv[1]
submission_filename = sys.argv[2]
labels_filename = sys.argv[3]

annotations = {}
with open(annotations_filename, 'r') as f:
    for line in f:
        s = line.strip().split('\t')
        annotations[s[0]] = s[1:]

labels = {}
with open(labels_filename, 'rb') as f:
    for line in f:
        ex = json.loads(line)
        if ex['gold_label'] == '-':
            continue
        else:
            labels[ex['pairID']] = ex['gold_label']

correct_guesses_by_annotation = defaultdict(float)
guesses_by_annotation = defaultdict(float)
correct_guesses = 0.0
guesses = 0.0
with open(submission_filename, 'r') as f:
    for line in f:
        s = line.strip().split(',')
        pairID = s[0]
        if pairID == 'pairID':
            continue

        guess = s[1]
        guesses += 1.0
        if guess == labels[pairID]:
            correct_guesses += 1.0

        if pairID in annotations:
            for annotation in annotations[pairID]:
                guesses_by_annotation[annotation] += 1.0
                if guess == labels[pairID]:
                    correct_guesses_by_annotation[annotation] += 1.0

print "Tag\tAccuracy\tInstances of tag"
for annotation in guesses_by_annotation:
    print "{}\t{:.1%}\t{}".format(annotation, correct_guesses_by_annotation[annotation] / guesses_by_annotation[annotation], int(guesses_by_annotation[annotation]))   
print
print "Overall accuracy: {:.1%}".format(correct_guesses / guesses)
