'''
Author: Badri Adhikari, University of Missouri-St. Louis, 11-13-2019
File: Contains the metrics to evaluate predicted real-valued distances, binned-distances and contact maps
'''

import numpy as np
import tensorflow as tf
epsilon = tf.keras.backend.epsilon()

from dataio import *
from generator import *

def calculate_mae(PRED, YTRUE, pdb_list, length_dict):
    avg_mae_lr_topL5 = 0.0
    avg_mae_lr_topL = 0.0
    avg_mae_mlr_topL5 = 0.0
    avg_mae_mlr_topL = 0.0
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        P = np.full((L, L), 100.0)
        # Average the predictions from both triangles (optional)
        # This can improve MAE upto 6%
        for j in range(0, L):
            for k in range(j, L):
                P[j, k] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        p_dict = {}
        y_dict = {}
        for j in range(0, L):
            for k in range(j, L):
                p_dict[(j,k)] = P[j, k]
                y_dict[(j,k)] = Y[j, k]
        # Count all long-range distances below 10A
        countlrle10 = 0
        for pair in sorted(y_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            (p, q) = k
            if (abs(p-q) < 24):
                continue
            if (y_dict[k]) > 10:
                continue
            countlrle10 += 1
        # Top L medium- and long-range distances
        mae_mlr_topL = 0.0
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            (p, q) = k
            if (abs(p-q) < 12):
                continue
            x -= 1
            if x == 0:
                break
            mae_mlr_topL += abs(y_dict[k] - p_dict[k])
        mae_mlr_topL /= L
        # Top L/5 medium- and long-range distances
        mae_mlr_topL5 = 0.0
        x = int(L/5)
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            (p, q) = k
            if (abs(p-q) < 12):
                continue
            x -= 1
            if x == 0:
                break
            mae_mlr_topL5 += abs(y_dict[k] - p_dict[k])
        mae_mlr_topL5 /= int(L/5)
        # Top L long-range distances
        mae_lr_topL = 0.0
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            (p, q) = k
            if (abs(p-q) < 24):
                continue
            x -= 1
            if x == 0:
                break
            mae_lr_topL += abs(y_dict[k] - p_dict[k])
        mae_lr_topL /= L
        # Top L/5 long-range distances
        mae_lr_topL5 = 0.0
        x = int(L/5)
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            (p, q) = k
            if (abs(p-q) < 24):
                continue
            x -= 1
            if x == 0:
                break
            mae_lr_topL5 += abs(y_dict[k] - p_dict[k])
        mae_lr_topL5 /= int(L/5)
        # Average for all pdbs
        avg_mae_lr_topL5 += mae_lr_topL5
        avg_mae_lr_topL += mae_lr_topL
        avg_mae_mlr_topL5 += mae_mlr_topL5
        avg_mae_mlr_topL += mae_mlr_topL
        print('MAE for ' + str(i) + ' - ' + str(pdb_list[i]) + '  top_L5_lr = %.2f  top_L_lr = %.2f  top_L5_mlr = %.2f  top_L_mlr = %.2f  (LR < 10A = %d)' % (mae_lr_topL5, mae_lr_topL, mae_mlr_topL5, mae_mlr_topL, countlrle10) )
    print('Average MAE      : top_L5_lr = %.4f  top_L_lr = %.4f  top_L5_mlr = %.4f  top_L_mlr = %.4f' %
          (avg_mae_lr_topL5 / len(PRED[:, 0, 0, 0]),
           avg_mae_lr_topL / len(PRED[:, 0, 0, 0]),
           avg_mae_mlr_topL5 / len(PRED[:, 0, 0, 0]),
           avg_mae_mlr_topL / len(PRED[:, 0, 0, 0])))

# Convert distances to contact probabilities
def distance_to_contacts(distance_matrix):
    P = 4.0 / distance_matrix
    P[P > 1.0] = 1.0
    return P

def calculate_contact_precision_in_distances(PRED, YTRUE, pdb_list, length_dict):
    Y = np.copy(YTRUE)
    Y[ Y < 8.0] = True
    Y[ Y >= 8.0] = False
    calculate_contact_precision(distance_to_contacts(PRED), Y, pdb_list, length_dict)

def calculate_contact_precision(PRED, YTRUE, pdb_list, length_dict):
    avg_p_lr_topL5 = 0.0
    avg_p_lr_topL = 0.0
    avg_p_lr_topNc = 0.0
    avg_p_mlr_topL5 = 0.0
    avg_p_mlr_topL = 0.0
    avg_p_mlr_topNc = 0.0
    count_pdbs_with_lr = 0
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        P = np.full((L, L), 0.0)
        # Average the predictions from both triangles
        for j in range(0, L):
            for k in range(j, L):
                P[j, k] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        p_dict = {}
        for j in range(0, L):
            for k in range(j, L):
                p_dict[(j,k)] = P[j, k]

        # Check if there are 0 medium-long-range contacts
        total_true_mlr = 0
        for j in range(0, L):
            for k in range(j, L):
                if (abs(j-k) < 12):
                    continue
                if YTRUE[i, k, j]:
                    total_true_mlr += 1

        # Check if there are 0 long-range contacts
        total_true_lr = 0
        for j in range(0, L):
            for k in range(j, L):
                if (abs(j-k) < 24):
                    continue
                if YTRUE[i, k, j]:
                    total_true_lr += 1

        # Top Nc (all) medium- and long-range distances
        p_mlr_topNc = 0
        x = total_true_mlr
        total_predictions = 0
        for pair in reversed(sorted(p_dict.items(), key=lambda x: x[1])):
            (k, v) = pair
            (p, q) = k
            if (abs(p-q) < 12):
                continue
            if YTRUE[i, p, q]:
                p_mlr_topNc += 1
            total_predictions += 1
            x -= 1
            if x == 0:
                break
        p_mlr_topNc /= total_predictions
        # Top L medium- and long-range distances
        p_mlr_topL = 0
        x = L
        total_predictions = 0
        for pair in reversed(sorted(p_dict.items(), key=lambda x: x[1])):
            (k, v) = pair
            (p, q) = k
            if (abs(p-q) < 12):
                continue
            if YTRUE[i, p, q]:
                p_mlr_topL += 1
            total_predictions += 1
            x -= 1
            if x == 0:
                break
        p_mlr_topL /= total_predictions
        # Top L/5 medium- and long-range distances
        p_mlr_topL5 = 0
        x = int(L/5)
        total_predictions = 0
        for pair in reversed(sorted(p_dict.items(), key=lambda x: x[1])):
            (k, v) = pair
            (p, q) = k
            if (abs(p-q) < 12):
                continue
            if YTRUE[i, p, q]:
                p_mlr_topL5 += 1
            total_predictions += 1
            x -= 1
            if x == 0:
                break
        p_mlr_topL5 /= total_predictions

        p_lr_all = float('nan')
        p_lr_topL = float('nan')
        p_lr_topL5 = float('nan')
        if total_true_lr > 0:
            # Top Nc (all) long-range distances
            p_lr_topNc = 0
            x = total_true_lr
            total_predictions = 0
            for pair in reversed(sorted(p_dict.items(), key=lambda x: x[1])):
                (k, v) = pair
                (p, q) = k
                if (abs(p-q) < 24):
                    continue
                if YTRUE[i, p, q]:
                    p_lr_topNc += 1
                total_predictions += 1
                x -= 1
                if x == 0:
                    break
            p_lr_topNc /= total_predictions
            # Top L long-range distances
            p_lr_topL = 0
            x = L
            total_predictions = 0
            for pair in reversed(sorted(p_dict.items(), key=lambda x: x[1])):
                (k, v) = pair
                (p, q) = k
                if (abs(p-q) < 24):
                    continue
                if YTRUE[i, p, q]:
                    p_lr_topL += 1
                total_predictions += 1
                x -= 1
                if x == 0:
                    break
            p_lr_topL /= total_predictions
            # Top L/5 long-range distances
            p_lr_topL5 = 0
            x = int(L/5)
            total_predictions = 0
            for pair in reversed(sorted(p_dict.items(), key=lambda x: x[1])):
                (k, v) = pair
                (p, q) = k
                if (abs(p-q) < 24):
                    continue
                if YTRUE[i, p, q]:
                    p_lr_topL5 += 1
                total_predictions += 1
                x -= 1
                if x == 0:
                    break
            p_lr_topL5 /= total_predictions
            # Average for all pdbs
            avg_p_lr_topL5 += p_lr_topL5
            avg_p_lr_topL += p_lr_topL
            avg_p_lr_topNc += p_lr_topNc
            avg_p_mlr_topL5 += p_mlr_topL5
            avg_p_mlr_topL += p_mlr_topL
            avg_p_mlr_topNc += p_mlr_topNc
            count_pdbs_with_lr += 1
        print('Precision for ' + str(i) + ' - ' + str(pdb_list[i]) + '  top_L5_lr = %.4f  top_L_lr = %.4f  top_Nc_lr = %.4f  top_L5_mlr = %.4f  top_L_mlr = %.4f  top_Nc_mlr = %.4f (total_true_lr = %d  total_true_mlr = %d)' % (p_lr_topL5, p_lr_topL, p_lr_topNc, p_mlr_topL5, p_mlr_topL, p_mlr_topNc, total_true_lr, total_true_mlr) )
    print('Average Precision: top_L5_lr = %.2f  top_L_lr = %.2f  top_Nc_lr = %.2f  top_L5_mlr = %.2f  top_L_mlr = %.2f  top_Nc_mlr = %.2f' %
          (100.0 * avg_p_lr_topL5 / count_pdbs_with_lr,
           100.0 * avg_p_lr_topL / count_pdbs_with_lr,
           100.0 * avg_p_lr_topNc / count_pdbs_with_lr,
           100.0 * avg_p_mlr_topL5 / len(PRED[:, 0, 0, 0]),
           100.0 * avg_p_mlr_topL / len(PRED[:, 0, 0, 0]),
           100.0 * avg_p_mlr_topNc / len(PRED[:, 0, 0, 0])))

def eval_distance_predictions(my_model, my_list, my_length_dict, my_dir_features, my_dir_distance, pad_size, flag_plots, flag_save, LMAX):
    # Padded but full inputs/outputs
    my_generator = DistGenerator(my_list, my_dir_features, my_dir_distance, LMAX, pad_size, 1)
    P = my_model.predict_generator(my_generator, max_queue_size=10, verbose=1)
    P = 100.0 / (P + epsilon)
    # Remove padding, i.e. shift up and left by int(pad_size/2)
    P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
    Y = get_bulk_output_dist_maps(my_list, my_dir_distance, LMAX)

    if flag_plots:
        plot_four_pair_maps(Y, P, my_list, my_length_dict)

    print('')
    calculate_mae(P, Y, my_list, my_length_dict)

    print('')
    calculate_contact_precision_in_distances(P, Y, my_list, my_length_dict)

    if flag_save:
        print('')
        print('Save predictions..')
        for i in range(len(my_list)):
            L = my_length_dict[my_list[i]]
            pred = P[i, :L, :L]
            save_contacts_rr(my_list[i], distance_to_contacts(pred), my_dir_features + my_list[i] + '.pkl', '/tmp/' + my_list[i] + '.4bydist.rr')

def eval_contact_predictions(my_model, my_list, my_length_dict, my_dir_features, my_dir_distance, pad_size, flag_plots, flag_save, LMAX):
    # Padded but full inputs/outputs
    my_generator = ContactGenerator(my_list, my_dir_features, my_dir_distance, LMAX, pad_size, 1)
    P = my_model.predict_generator(my_generator, max_queue_size=10, verbose=1)
    # Remove padding, i.e. shift up and left by int(pad_size/2)
    P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
    Y = get_bulk_output_contact_maps(my_list, my_dir_distance, LMAX)

    if flag_plots:
        plot_four_pair_maps(Y, P, my_list, my_length_dict)

    print('')
    calculate_contact_precision(P, Y, my_list, my_length_dict)

    if flag_save:
        print('')
        print('Save predictions..')
        for i in range(len(my_list)):
            L = my_length_dict[my_list[i]]
            pred = P[i, :L, :L]
            save_contacts_rr(my_list[i], pred, my_dir_features + my_list[i] + '.pkl', '/tmp/' + my_list[i] + '.contacts.rr')

def eval_binned_predictions(my_model, my_list, my_length_dict, my_dir_features, my_dir_distance, pad_size, flag_plots, flag_save, LMAX, bins):
    # Padded but full inputs/outputs
    my_generator = BinnedDistGenerator(my_list, my_dir_features, my_dir_distance, bins, LMAX, pad_size, 1)
    P = my_model.predict_generator(my_generator, max_queue_size=10, verbose=1)
    # Remove padding, i.e. shift up and left by int(pad_size/2)
    P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
    Y = get_bulk_output_contact_maps(my_list, my_dir_distance, LMAX)

    # Predicted distance is mean distance of the most confident bin
    D = np.zeros((len(P), LMAX, LMAX, 1))
    for p in range(len(P)):
        for i in range(LMAX):
            for j in range(LMAX):
                index = np.argmax(P[p, i, j, :])
                min_max = [float(x) for x in bins[index].split()]
                D[p, i, j, 0] = ( min_max[0] + min_max[1] ) / 2.0

    # The last bin's range has a very large value, so trim it
    bin_max = float(bins[len(bins) - 1].split()[0])
    D[D > bin_max] = bin_max

    Y = get_bulk_output_dist_maps(my_list, my_dir_distance, LMAX)

    print('')
    calculate_mae(D, Y, my_list, my_length_dict)

    # Identify the bins that fall under the 8.0A distance
    contact_bins = -1
    for k, v in bins.items():
        if bins[k].split()[0] == '8.0':
            contact_bins = k

    # Sum the probabilities of the bins that fall under 8.0A distance
    C = np.zeros((len(P), LMAX, LMAX, 1))
    for p in range(len(P)):
        for i in range(LMAX):
            for j in range(LMAX):
                C[p, i, j, 0] = np.sum(P[p, i, j, :contact_bins])

    Y = get_bulk_output_contact_maps(my_list, my_dir_distance, LMAX)

    print('')
    calculate_contact_precision(C, Y, my_list, my_length_dict)

    if flag_save:
        print('')
        print('Save predictions..')
        for i in range(len(my_list)):
            L = my_length_dict[my_list[i]]
            predictions = {}
            for b in range(len(bins)):
                predictions[bins[b]] = P[i, :L, :L, b].astype(np.float16)
            f = open('/tmp/' + my_list[i] + '.bins.pkl', 'wb')
            pickle.dump(predictions, f)
            f.close()
