# Converts into pairs of samples for pairwise log training.

import numpy as np
import pandas as bp
from sklearn.cross_validation import train_test_split
from itertools import groupby, chain
import random

def stringReader(filename, max_lines):
    lineno = 0
    for line in open(filename):
        lineno += 1
        if lineno % 1000000 == 0:
            print lineno, " lines processed"
        if max_lines and lineno > max_lines:
            break
        yield [float(i) for i in line.split(',')]

def outputSampleConcat(outfile, label, sample_1, sample_2):
    """ Outputs a string with concatinated features.
    """
    outfile.write(label)
    outfile.write(',')
    outfile.write(",".join([str(i) for i in chain(sample_1, sample_2)]))
    outfile.write('\n')

def outputSampleDiff(outfile, label, sample_1, sample_2):
    outfile.write(label)
    outfile.write(',')
    outfile.write(",".join([str(x1-x2) for x1,x2 in zip(sample_1, sample_2)]))
    outfile.write('\n')

def randomArrayElement(array_in):
    return array_in[random.randint(0, len(array_in) - 1)]


def processFileInt(infile_name, outfile_train_name, outfile_test_name,
        split_ratio,  max_lines,  sampleOutputer):
    outfile_train = open(outfile_train_name, "w")
    outfile_test = open(outfile_test_name, "w")
    for k,g in groupby(stringReader(infile_name, max_lines), lambda x: x[1]):
        positive_samples, negative_samples = [],[]
        for r in g:
            if r[0] > 0:
                positive_samples.append(r)
            else:
                negative_samples.append(r)
        if not positive_samples or not negative_samples:
            continue
        outfile = outfile_test if random.random() < split_ratio else outfile_train
        used_pairs = set()
        # sampling starts
        for i in range(len(positive_samples)+len(negative_samples)):
            positive_index = random.randint(0, len(positive_samples) - 1)
            negative_index = random.randint(0, len(negative_samples) - 1)
            if (positive_index, negative_index) not in used_pairs:
                used_pairs.add((positive_index, negative_index))
                positive_sample = randomArrayElement(positive_samples)
                negative_sample = randomArrayElement(negative_samples)
                if random.random() > 0.5:
                    sampleOutputer(outfile, '1', positive_sample[3:], negative_sample[3:])
                else:
                    sampleOutputer(outfile, '0', negative_sample[3:], positive_sample[3:])
    outfile_train.close()
    outfile_test.close()

def processFileConcatFeatures(infile_name, outfile_train_name, 
        outfile_test_name, split_ratio, max_lines):
    processFileInt(infile_name, outfile_train_name, outfile_test_name,
            split_ratio, max_lines, outputSampleConcat)


def processFileDiffFeatures(infile_name, outfile_name, max_lines):
    processFileInt(infile_name, outfile_name, max_lines, outputSampleDiff)

def LoadPairsConcatDataset(name, test_size):
    """
    Load a dataset into memory. Splits into training/testing according to split
    factor. 
    Returns:
    X_train, X_test, y_train, y_test
    """
    data=bp.read_csv(name, header=None)
    return train_test_split(data[range(1,57)].values, data[0].values,
            test_size=test_size)

def LoadPairsDiffDataset(name, test_size):
    """
    Load a dataset into memory. Splits into training/testing according to split
    factor. 
    Returns:
    X_train, X_test, y_train, y_test
    """
    data=bp.read_csv(name, header=None)
    return train_test_split(data[range(1,29)].values, data[0].values,
            test_size=test_size)


