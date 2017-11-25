import dataset_parsers as dp
import os


filename = "/home/iindyk/PycharmProjects/AdversarialVirusDetection/dumps/harmless/0.txt"
dictionary = ['push', 'mov', 'sub']
dp.files2freq_pickle(os.getcwd()+'/dumps', dictionary, 0.2, 0.2)
