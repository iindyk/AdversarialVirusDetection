import dataset_parsers as dp


filename = "/home/iindyk/PycharmProjects/AdversarialVirusDetection/dumps/harmless/0.txt"
dictionary = ['push', 'mov', 'sub']
print(dp.file2freq(filename, dictionary))