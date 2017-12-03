import os
import pickle
import numpy as np


# convert opcode txt file to dictionary of frequencies
def file2freq(filename, dictionary):
    file = open(filename, 'r')
    count = np.zeros(len(dictionary))
    words = file.read().split()

    for word in words:
        if word in dictionary:
            count[dictionary.index(word)] += 1
    file.close()
    return count/sum(count)


# convert subfolders "harmless" and "viruses" in current folder to dataset pickle file
def files2freq_pickle(folder_name, dictionary, valid_share, test_share):
    n = len(os.listdir(folder_name+'/harmless'))+len(os.listdir(folder_name+'/viruses'))
    m = len(dictionary)
    assert 1.0 > valid_share + test_share

    dataset = np.zeros((n, m))
    labels = np.zeros(n)

    # harmless files
    i = 0
    for f in os.listdir(folder_name+'/harmless'):
        dataset[i, :] = file2freq(folder_name+'/harmless/'+f, dictionary)
        labels[i] = 1.0
        i += 1
    n_harmless = i

    # viruses
    for v in os.listdir(folder_name+'/viruses'):
        dataset[i, :] = file2freq(folder_name+'/viruses/'+v, dictionary)
        labels[i] = -1.0
        i += 1

    # split dataset into train, validation and test datasets
    idx_harmless = np.arange(n_harmless)
    np.random.shuffle(idx_harmless)
    idx_viruses = np.arange(n_harmless, n)
    np.random.shuffle(idx_viruses)
    val_v_idx, test_v_idx, train_v_idx = \
        np.split(idx_viruses, [int(valid_share * (n-n_harmless)),
                               int((valid_share + test_share) * (n-n_harmless))])
    train_dataset = dataset[train_v_idx, :]
    train_labels = labels[train_v_idx]
    valid_dataset = dataset[val_v_idx, :]
    valid_labels = labels[val_v_idx]
    test_dataset = dataset[test_v_idx, :]
    test_labels = labels[test_v_idx]
    val_h_idx, test_h_idx, train_h_idx = \
        np.split(idx_harmless, [int(valid_share * n_harmless),
                                int((valid_share + test_share) * n_harmless)])
    train_dataset = np.append(train_dataset, dataset[train_h_idx, :], axis=0)
    train_labels = np.append(train_labels, labels[train_h_idx], axis=0)
    valid_dataset = np.append(valid_dataset, dataset[val_h_idx, :], axis=0)
    valid_labels = np.append(valid_labels, labels[val_h_idx], axis=0)
    test_dataset = np.append(test_dataset, dataset[test_h_idx, :], axis=0)
    test_labels = np.append(test_labels, labels[test_h_idx], axis=0)

    # save dataset to pickle
    dataset_dict = {'train_dataset': train_dataset, 'train_labels': train_labels,
                    'valid_dataset': valid_dataset, 'valid_labels': valid_labels,
                    'test_dataset': test_dataset, 'test_labels': test_labels}
    pickle.dump(dataset_dict, open("dataset.p", "wb"))


# read dataset from pickle
def read_data():
    dataset_dict = pickle.load(open("dataset.p", "rb"))
    train_dataset = dataset_dict['train_dataset']
    train_labels = dataset_dict['train_labels']
    valid_dataset = dataset_dict['valid_dataset']
    valid_labels = dataset_dict['valid_labels']
    test_dataset = dataset_dict['test_dataset']
    test_labels = dataset_dict['test_labels']
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


# create dictionary of used opcode commands
def create_dictionary():
    dictionary = {}
    directory_in_str = '/home/iindyk/PycharmProjects/AdversarialVirusDetection/dumps/test'
    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        f = open(filename, 'r')



