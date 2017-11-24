import os
import numpy as np


# convert opcode txt file to dictionary of frequencies
def file2freq(filename, dictionary):
    file = open(filename, 'r')
    counts = np.zeros(len(dictionary))
    words = file.read().split()

    for word in words:
        if word in dictionary:
            counts[dictionary.index(word)] += 1
    file.close()
    return counts


# convert subfolders "harmless" and "viruses" in current folder to dataset pickle file
def files2freq_pickle(folder_name, dictionary):
    dataset = np.array((len(os.listdir(folder_name)), len(dictionary)))
    labels = np.zeros(len(os.listdir(folder_name)))

    # harmless files
    i = 0
    for f in os.listdir(folder_name+'/harmless'):
        dataset[i] = file2freq(f, dictionary)
        labels[i] = 1.0
    print(dataset)
    print(labels)



