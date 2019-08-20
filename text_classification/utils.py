import os
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from IPython.display import clear_output
from sklearn.preprocessing import LabelEncoder


def char_distribution(phrases):
    val = phrases.values
    di = Counter(val.ravel().sum())
    labels, values = zip(*di.most_common())
    indexes = np.arange(len(labels))
    values = np.array(values)
    values = values / values.sum()
    width = 1

    plt.figure(figsize=(25, 10))
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.title("{} znaków".format(len(di)))
    plt.show()
    return di


def load_data(folder_path, labels):
    data = []
    for folder_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(class_path):
            for text_file_name in os.listdir(class_path):
                if text_file_name.endswith("txt"):
                    text = open(os.path.join(
                        class_path,
                        text_file_name
                    ), "rb").read().decode('latin')
                    text = re.sub(r"\W+", " ", text)
                    text = re.sub(r" +", " ", text)
                    data.append((text, folder_name))
    
    data_df = pd.DataFrame(data, columns=labels)
    encoder = LabelEncoder()
    data_df[labels[1]] = encoder.fit_transform(data_df[labels[1]])
                    
    return data_df, encoder


def get_proper_callback(callback_class, arg=None):
    class ProperCallback(callback_class):
        def __init__(self, arg=None):
            if arg:
                super(ProperCallback, self).__init__(**arg)
            else:
                super(ProperCallback, self).__init__()
            
        def on_train_batch_begin(self, batch, logs=None):
            return super(ProperCallback, self).on_batch_begin(batch, logs)

        def on_train_batch_end(self, batch, logs=None):
            return super(ProperCallback, self).on_batch_end(batch, logs)

        def on_test_batch_begin(self, batch, logs=None):
            print('Evaluating: batch {} begin at {}'.format(batch, datetime.datetime.now().time()))

        def on_test_batch_end(self, batch, logs=None):
            clear_output(wait=True)
            print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
    
    return ProperCallback(arg)




