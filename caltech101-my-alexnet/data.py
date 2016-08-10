import pickle
import sys, os

import numpy as np
from PIL import Image

path_prefix = 'caltech101_crop227'

num_train = 0
num_test = 0

# first
def loadCaltech101Data():
    print('Loading images...')
    images = []
    labels = []
    label_names = []
    label_i = 0
    for item in os.listdir(path_prefix):
        if not os.path.isdir(path_prefix + '/' + item): continue
        label = item
        label_names.append(label)
        for jpg in os.listdir(path_prefix + '/' + item):
            if jpg.find('.jpg') != -1 or jpg.find('jpeg') != -1:
                # load jpeg file
                im = np.array(Image.open(path_prefix + '/' + label + '/' + jpg).convert('RGB')).transpose((2,0,1))
                if im.shape != (3,227,227):
                    print('skip {}/{}'.format(label, jpg))
                    continue
                images.append(im)
                labels.append(label_i)
        label_i += 1
        print('.', end='')
        sys.stdout.flush()
    print('Done. (loaded %d images)' % len(images))

    x_train = np.asarray(images)
    y_train = np.asarray(labels)
    print(x_train.shape)
    caltech = {'x_train': x_train, 'y_train': y_train, 'label_names': label_names}

    print('Save output...')
    with open('caltech101.pkl', 'wb') as output:
        pickle.dump(caltech, output, -1)
    print('done')

def loadCaltech101():
    if not os.path.exists('caltech101.pkl'):
        loadCaltech101Data()

    with open('caltech101.pkl', 'rb') as caltech_pickle:
        caltech = pickle.load(caltech_pickle)

    global num_train
    num_train = caltech['x_train'].shape[0]
    return caltech

