import os
import pickle
import numpy as np
from random import uniform, randint
from PIL import Image


# convert opcode txt file to dictionary of frequencies
def file2freq(filename, dictionary):
    file = open(filename, 'r')
    count = np.zeros(len(dictionary))
    words = file.read().split()

    for word in words:
        if not (any(char.isdigit() for char in word) or '%' in word or '<' in word or '(' in word or '.' in word
                or ':' in word) and word in dictionary:
            count[dictionary.index(word)] += 1
    file.close()
    if sum(count) == 0:
        return count
    else:
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
        print(i)
        try:
            dataset[i, :] = file2freq(folder_name+'/harmless/'+f, dictionary)
        except UnicodeDecodeError:
            continue
        labels[i] = 1.0
        i += 1
    n_harmless = i

    # viruses
    for v in os.listdir(folder_name+'/viruses'):
        print(i)
        try:
            dataset[i, :] = file2freq(folder_name+'/viruses/'+v, dictionary)
        except UnicodeDecodeError:
            continue
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

    # shuffle datasets
    train_idx_s = np.arange(len(train_labels))
    np.random.shuffle(train_idx_s)
    train_dataset_s = train_dataset[train_idx_s,:]
    train_labels_s = train_labels[train_idx_s]
    valid_idx_s = np.arange(len(valid_labels))
    np.random.shuffle(valid_idx_s)
    valid_dataset_s = valid_dataset[valid_idx_s, :]
    valid_labels_s = valid_labels[valid_idx_s]
    test_idx_s = np.arange(len(test_labels))
    np.random.shuffle(test_idx_s)
    test_dataset_s = test_dataset[test_idx_s, :]
    test_labels_s = test_labels[test_idx_s]

    # save dataset to pickle
    dataset_dict = {'train_dataset': train_dataset_s, 'train_labels': train_labels_s,
                    'valid_dataset': valid_dataset_s, 'valid_labels': valid_labels_s,
                    'test_dataset': test_dataset_s, 'test_labels': test_labels_s}
    pickle.dump(dataset_dict, open("/home/iindyk/PycharmProjects/AdversarialVirusDetection/data/dataset.p", "wb"))


# read dataset from pickle
def read_data():
    dataset_dict = pickle.load(open("/home/iindyk/PycharmProjects/AdversarialVirusDetection/data/dataset.p", "rb"))
    train_dataset_bad = dataset_dict['train_dataset']
    train_labels_bad = dataset_dict['train_labels']
    train_zeros = np.where(train_labels_bad == 0)
    train_dataset = np.delete(train_dataset_bad, train_zeros, axis=0)
    train_labels = np.delete(train_labels_bad, train_zeros)
    valid_dataset_bad = dataset_dict['valid_dataset']
    valid_labels_bad = dataset_dict['valid_labels']
    valid_zeros = np.where(valid_labels_bad == 0)
    valid_dataset = np.delete(valid_dataset_bad, valid_zeros, axis=0)
    valid_labels = np.delete(valid_labels_bad, valid_zeros)
    test_dataset_bad = dataset_dict['test_dataset']
    test_labels_bad = dataset_dict['test_labels']
    test_zeros = np.where(test_labels_bad == 0)
    test_dataset = np.delete(test_dataset_bad, test_zeros, axis=0)
    test_labels = np.delete(test_labels_bad, test_zeros)
    return train_dataset[:, :20], train_labels, valid_dataset[:, :20], valid_labels, test_dataset[:, :20], test_labels


# create dictionary of used opcode commands
def create_dictionary():
    dictionary = {}

    directory_in_str_v = '/home/iindyk/PycharmProjects/AdversarialVirusDetection/data/dumps/viruses'
    for file in os.listdir(os.fsencode(directory_in_str_v)):
        filename = os.fsdecode(file)
        f = open(directory_in_str_v+'/'+filename, 'r')
        try:
            words = f.read().split()
        except UnicodeDecodeError:
            continue
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            elif not (any(char.isdigit() for char in word) or '%' in word or '<' in word or '(' in word or '.' in word
                      or ':' in word):
                dictionary[word] = 1
        f.close()

    directory_in_str_h = '/home/iindyk/PycharmProjects/AdversarialVirusDetection/data/dumps/harmless'
    for file in os.listdir(os.fsencode(directory_in_str_h)):
        filename = os.fsdecode(file)
        f = open(directory_in_str_h + '/' + filename, 'r')
        try:
            words = f.read().split()
        except UnicodeDecodeError:
            continue
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            elif not (any(char.isdigit() for char in word) or '%' in word or '<' in word or '(' in word or '.' in word
                      or ':' in word):
                dictionary[word] = 1
        f.close()

    final_dict = list(dictionary.keys())
    for word in dictionary:
        if dictionary[word] < 20:
            final_dict.remove(word)
    final_dict.remove('file')
    final_dict.remove('format')
    final_dict.remove('Disassembly')
    final_dict.remove('of')
    final_dict.remove('section')
    final_dict.remove('ff')
    final_dict.remove('fc')
    final_dict.remove('ec')
    final_dict.remove('da')
    final_dict.append('int3')

    dictionary_file = open('/home/iindyk/PycharmProjects/AdversarialVirusDetection/data/dictionary.txt', 'w')
    for word in final_dict:
        dictionary_file.write("%s\n" % word)
    dictionary_file.close()
    print(len(final_dict))
    print(final_dict)


def get_toy_dataset(n, m):
    dataset = np.random.uniform(0, 1, (n, m))
    labels = []
    for i in range(n):
        if sum(dataset[i, :]) > 0.5*m:
            labels.append(1)
        else:
            labels.append(-1)
    # random attack
    for i in range(int(0.1*n)):
        k = randint(0, n-1)
        if labels[k] == 1:
            labels[k] = -1
        else:
            labels[k] = 1
    return dataset[:int(0.5*n), :], labels[:int(0.5*n)], dataset[int(0.5*n):, :], labels[int(0.5*n):]


def read_data_cat_dog():
    dir_tr = 'data/images/cat_dog/train'
    data = []
    labels = []
    size = 64, 64
    maxit = 2000
    nit = 0

    for file in os.listdir(dir_tr):
        filename = os.fsdecode(file)
        img = Image.open(os.path.join(dir_tr, filename)).convert('RGBA')
        img = img.resize(size, Image.ANTIALIAS)
        arr = np.array(img)

        data.append(arr.ravel())
        if 'dog' in file:
            labels.append(-1.0)
        else:
            labels.append(1.0)

        nit += 1
        # print(nit)
        if nit > maxit:
            break

    return data[:1500], labels[:1500], data[1500:1700], labels[1500:1700], data[1700:], labels[1700:]




