'''
Author: Badri Adhikari, University of Missouri-St. Louis,  12-30-2019
File: Contains the code to train and test learning real-valued distances, binned-distances and contact maps
'''

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import sys
import numpy as np
import datetime
#%matplotlib inline

sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])) + '/lib')

from dataio import *
from metrics import *
from generator import *
from models import *
from losses import *

flag_plots = False

if flag_plots:
    from plots import *

if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)

# Some GPUs don't allow memory growth by default (keep both options)
# Option 1
#for gpu in tf.config.experimental.list_physical_devices('GPU'):
#    tf.config.experimental.set_memory_growth(gpu, True)
# Option 2
import keras.backend as K
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.tensorflow_backend.set_session(sess)

if len(sys.argv) != 10:
    print('')
    print('Insufficient arguments!!')
    print('')
    print('Usage:')
    print(' > python3 ' + sys.argv[0] + ' <contact/distance/binned> <wts-file> <training-size> <train-window> <training_epochs> <arch-arch_depth> <filters-per-layer> <dir-data> <flag-evaluate>')
    print('')
    print('Example:')
    print(' > python3 ' + sys.argv[0] + ' distance wts.dist.hdf5 -1 64 4 8 64 ./ 0')
    print('')
    sys.exit(1)

contact_or_dist_or_binned = sys.argv[1]      # contact or distance or binnned
file_weights              = sys.argv[2]
num_chains_for_training   = int(sys.argv[3]) # -1 loads all data
training_window           = int(sys.argv[4])
training_epochs           = int(sys.argv[5])
arch_depth                = int(sys.argv[6])
filters_per_layer         = int(sys.argv[7])
dir_dataset               = sys.argv[8]
flag_evaluate_only        = int(sys.argv[9]) # 0/1
pad_size                  = 10
batch_size                = 2
expected_n_channels       = 882+55+85

print('Start ' + str(datetime.datetime.now()))

print('')
print('Parameters:')
print('contact_or_dist_or_binned', contact_or_dist_or_binned)
print('num_chains_for_training', num_chains_for_training)
print('file_weights', file_weights)
print('training_window', training_window)
print('training_epochs', training_epochs)
print('arch_depth', arch_depth)
print('filters_per_layer', filters_per_layer)
print('pad_size', pad_size)
print('batch_size', batch_size)
print('dir_dataset', dir_dataset)

if not (contact_or_dist_or_binned == 'distance' or contact_or_dist_or_binned == 'contact' or contact_or_dist_or_binned == 'binned'):
    print('ERROR! Invalid input choice!')
    sys.exit(1)

bins = {}
if contact_or_dist_or_binned == 'binned':
    bins[0] = '0.0 4.0'
    b = 1
    range_min = float(bins[0].split()[1])
    interval = 0.2
    while(range_min < 8.0):
        bins[b] = str(round(range_min, 2)) + ' ' + str(round(range_min + interval, 2))
        range_min += interval
        b += 1
    while(range_min <= 26):
        interval += 0.2
        bins[b] = str(round(range_min, 2)) + ' ' + str(round(range_min + interval, 2))
        b += 1
        range_min += interval
    bins[b] = '26.0 1000.0'
    print('')
    print('Number of bins', len(bins))
    print('Actual bins:', bins)

deepcov_list = load_list(dir_dataset + '/deepcov.lst', num_chains_for_training)

length_dict = {}
for pdb in deepcov_list:
    (ly, seqy, cb_map) = np.load(dir_dataset + '/deepcov/distance/' + pdb + '-cb.npy', allow_pickle = True)
    length_dict[pdb] = ly

print('')
print('Split into training and validation set (4%)..')
split = int(0.0377 * len(deepcov_list))
valid_pdbs = deepcov_list[:split]
train_pdbs = deepcov_list[split:]



print('Total validation proteins : ', len(valid_pdbs))
print('Total training proteins   : ', len(train_pdbs))

train_generator = ''
valid_generator = ''
if contact_or_dist_or_binned == 'distance':
    train_generator = DistGenerator(train_pdbs, dir_dataset + '/ssd2tb/pre441c/', dir_dataset + '/deepcov/distance/', training_window, pad_size, batch_size)
    valid_generator = DistGenerator(valid_pdbs, dir_dataset + '/ssd2tb/pre441c/', dir_dataset + '/deepcov/distance/', training_window, pad_size, batch_size)
if contact_or_dist_or_binned == 'contact':
    train_generator = ContactGenerator(train_pdbs, dir_dataset + '/ssd2tb/', dir_dataset + '/deepcov/distance/', training_window, pad_size, batch_size)
    #train_generator = ContactGenerator(train_pdbs, dir_dataset + '/ssd2tb/cov231/', dir_dataset + '/deepcov/distance/', training_window, pad_size, batch_size)
    valid_generator = ContactGenerator(valid_pdbs, dir_dataset + '/ssd2tb/', dir_dataset + '/deepcov/distance/', training_window, pad_size, batch_size)
    #valid_generator = ContactGenerator(valid_pdbs, dir_dataset + '/ssd2tb/cov231/', dir_dataset + '/deepcov/distance/', training_window, pad_size, batch_size)
if contact_or_dist_or_binned == 'binned':
    train_generator = BinnedDistGenerator(train_pdbs, dir_dataset + '/ssd2tb/pre441c/', dir_dataset + '/deepcov/distance/', bins, training_window, pad_size, batch_size)
    valid_generator = BinnedDistGenerator(valid_pdbs, dir_dataset + '/ssd2tb/pre441c/', dir_dataset + '/deepcov/distance/', bins, training_window, pad_size, batch_size)

print('')
print('len(train_generator) : ' + str(len(train_generator)))
print('len(valid_generator) : ' + str(len(valid_generator)))

X, Y = train_generator[1]
print('Actual shape of X    : ' + str(X.shape))
print('Actual shape of Y    : ' + str(Y.shape))

print('')
print('Channel summaries:')
summarize_channels(X[0, :, :, :], Y[0])

if flag_plots:
    print('')
    print('Inputs/Output of protein', 0)
    plot_protein_io(X[0, :, :, :], Y[0, :, :, 0])

print('')
print('Build a model..')
model = ''
if contact_or_dist_or_binned == 'distance':
    model = deepcon_rdd_distances(training_window, arch_depth, filters_per_layer, expected_n_channels)
if contact_or_dist_or_binned == 'contact':
    model = deepcon_rdd(training_window, arch_depth, filters_per_layer, expected_n_channels)
if contact_or_dist_or_binned == 'binned':
    model = deepcon_rdd_binned(training_window, arch_depth, filters_per_layer, len(bins), expected_n_channels)

print('')
print('Compile model..')
if contact_or_dist_or_binned == 'distance':
    model.compile(loss = 'logcosh', optimizer = 'rmsprop', metrics = ['mae'])
if contact_or_dist_or_binned == 'contact':
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
if contact_or_dist_or_binned == 'binned':
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

print(model.summary())

if flag_evaluate_only == 0:
    if os.path.exists(file_weights):
        print('')
        print('Loading existing weights..')
        model.load_weights(file_weights)

    print('')
    print('Train..')

    history = model.fit_generator(generator = train_generator,
        validation_data = valid_generator,
        callbacks = [ModelCheckpoint(filepath = file_weights, monitor = 'val_loss', save_best_only = True, save_weights_only = True, verbose = 1)],
        verbose = 1,
        max_queue_size = 8,
        workers = 1,
        use_multiprocessing = False,
        shuffle = True ,
        epochs = training_epochs)

    if flag_plots:
        plot_learning_curves(history)

LMAX = 512

print('')
print('Evaluate validation set..')

if contact_or_dist_or_binned == 'distance':
    model = deepcon_rdd_distances(LMAX, arch_depth, filters_per_layer, expected_n_channels)
    model.load_weights(file_weights)
    eval_distance_predictions(model, valid_pdbs, length_dict, dir_dataset + '/ssd2tb/pre441c/', dir_dataset + '/deepcov/distance/', pad_size, flag_plots, False, LMAX)

if contact_or_dist_or_binned == 'contact':
    model = deepcon_rdd(LMAX, arch_depth, filters_per_layer, expected_n_channels)
    model.load_weights(file_weights)
    eval_contact_predictions(model, valid_pdbs, length_dict, dir_dataset + '/ssd2tb/', dir_dataset + '/deepcov/distance/', pad_size, flag_plots, False, LMAX)
    #eval_contact_predictions(model, valid_pdbs, length_dict, dir_dataset + '/ssd2tb/cov231/', dir_dataset + '/deepcov/distance/', pad_size, flag_plots, False, LMAX)

if contact_or_dist_or_binned == 'binned':
    model = deepcon_rdd_binned(LMAX, arch_depth, filters_per_layer, len(bins), expected_n_channels)
    model.load_weights(file_weights)
    eval_binned_predictions(model, valid_pdbs, length_dict, dir_dataset + '/ssd2tb/pre441c/', dir_dataset + '/deepcov/distance/', pad_size, flag_plots, False, LMAX, bins)

print('')
print('Evaluate test set..')

psicov_list = load_list(dir_dataset + 'psicov.lst')
length_dict = {}
for pdb in psicov_list:
    (ly, seqy, cb_map) = np.load(dir_dataset + '/psicov/distance/' + pdb + '-cb.npy', allow_pickle = True)
    length_dict[pdb] = ly

if contact_or_dist_or_binned == 'distance':
    model = deepcon_rdd_distances(LMAX, arch_depth, filters_per_layer, expected_n_channels)
    model.load_weights(file_weights)
    eval_distance_predictions(model, psicov_list, length_dict, dir_dataset + '/psicov/cov231/', dir_dataset + '/psicov/distance/', pad_size, flag_plots, True, LMAX)

if contact_or_dist_or_binned == 'contact':
    model = deepcon_rdd(LMAX, arch_depth, filters_per_layer, expected_n_channels)
    model.load_weights(file_weights)
    eval_contact_predictions(model, psicov_list, length_dict, dir_dataset + '/psicov/cov231/', dir_dataset + '/psicov/distance/', pad_size, flag_plots, False, LMAX)

if contact_or_dist_or_binned == 'binned':
    model = deepcon_rdd_binned(LMAX, arch_depth, filters_per_layer, len(bins), expected_n_channels)
    model.load_weights(file_weights)
    eval_binned_predictions(model, psicov_list, length_dict, dir_dataset + '/psicov/cov16bit/', dir_dataset + '/psicov/distance/', pad_size, flag_plots, True, LMAX, bins)

print('')
print ('Everything done! ' + str(datetime.datetime.now()) )
