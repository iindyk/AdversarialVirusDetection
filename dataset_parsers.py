import os
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
    print('train dataset: ', train_dataset, train_labels)
    print('valid dataset: ', valid_dataset, valid_labels)
    print('test dataset: ', test_dataset, test_labels)



