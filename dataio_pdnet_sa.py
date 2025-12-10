'''
Author: Badri Adhikari, University of Missouri-St. Louis, 12-18-2019
File: Contains subroutines to load input features and output maps
'''

import pickle
import random
import numpy as np
import tensorflow as tf
import numpy as np
epsilon = tf.keras.backend.epsilon()

def load_list(file_lst, max_items = 1000000):
    if max_items < 0:
        max_items = 1000000
    protein_list = []
    f = open(file_lst, 'r')
    for l in f.readlines():
        protein_list.append(l.strip().split()[0])
    if (max_items < len(protein_list)):
        protein_list = protein_list[:max_items]
    return protein_list

def summarize_channels(x, y):
    print(' Channel        Avg        Max        Sum')
    for i in range(len(x[0, 0, :])):
        (m, s, a) = (x[:, :, i].flatten().max(), x[:, :, i].flatten().sum(), x[:, :, i].flatten().mean())
        print(' %7s %10.4f %10.4f %10.1f' % (i+1, a, m, s))
    print("      Ymin = %.2f  Ymean = %.2f  Ymax = %.2f" % (y.min(), y.mean(), y.max()) )

def get_bulk_output_contact_maps(pdb_id_list, distmap_path, OUTL):
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        Y = get_map(pdb, distmap_path)
        ly = len(Y[:, 0])
        assert ly <= OUTL
        YY[i, :ly, :ly, 0] = Y
    assert not np.any(np.isnan(Y))
    YY[ YY < 8.0 ] = 1.0
    YY[ YY >= 8.0 ] = 0.0
    return YY.astype(np.float32)

def get_bulk_output_dist_maps(pdb_id_list, distmap_path, OUTL):
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        Y = get_map(pdb, distmap_path)
        ly = len(Y[:, 0])
        assert ly <= OUTL
        YY[i, :ly, :ly, 0] = Y
    assert not np.any(np.isnan(Y))
    return YY.astype(np.float32)

def get_input_output_dist(pdb_id_list, features_path, distmap_path, pad_size, OUTL, expected_n_channels = 882+55+85):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0)
    YY = np.full((len(pdb_id_list), OUTL, OUTL, 1), 100.0)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, features_path, expected_n_channels)
        assert len(X[0, 0, :]) == expected_n_channels
        Y = get_map(pdb, distmap_path, len(X[:, 0, 0]))
        assert len(X[:, 0, 0]) == len(Y[:, 0])
        l = len(X[:, 0, 0])
        Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])), dtype=np.float32)
        Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
        Ypadded = np.full((l + pad_size, l + pad_size), 100.0, dtype=np.float32)
        Ypadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2)] = Y
        l = len(Xpadded[:, 0, 0])
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, 0] = Ypadded
        else:
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            assert rx + OUTL <= l
            assert ry + OUTL <= l
            XX[i, :, :, :] = Xpadded[rx:rx+OUTL, ry:ry+OUTL, :]
            YY[i, :, :, 0] = Ypadded[rx:rx+OUTL, ry:ry+OUTL]
    return XX.astype(np.float32), YY.astype(np.float32)

def get_input_output_bins(pdb_id_list, features_path, distmap_path, pad_size, OUTL, bins, expected_n_channels = 55):
    XX = np.full((len(pdb_id_list), OUTL, OUTL, expected_n_channels), 0.0)
    YY = np.full((len(pdb_id_list), OUTL, OUTL, len(bins)), 0.0)
    for i, pdb in enumerate(pdb_id_list):
        X = get_feature(pdb, features_path, expected_n_channels)
        assert len(X[0, 0, :]) == expected_n_channels
        Y = dist_map_to_bins(get_map(pdb, distmap_path, len(X[:, 0, 0])), bins)
        assert len(X[:, 0, 0]) == len(Y[:, 0])
        l = len(X[:, 0, 0])
        Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])))
        Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
        Ypadded = np.full((l + pad_size, l + pad_size, len(bins)), 0.0)
        Ypadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = Y
        l = len(Xpadded[:, 0, 0])
        if l <= OUTL:
            XX[i, :l, :l, :] = Xpadded
            YY[i, :l, :l, :] = Ypadded
        else:
            rx = random.randint(0, l - OUTL)
            ry = random.randint(0, l - OUTL)
            assert rx + OUTL <= l
            assert ry + OUTL <= l
            XX[i, :, :, :] = Xpadded[rx:rx+OUTL, ry:ry+OUTL, :]
            YY[i, :, :, :] = Ypadded[rx:rx+OUTL, ry:ry+OUTL, :]
    return XX.astype(np.float32), YY.astype(np.float32)

def get_sequence(pdb, feature_file):
    features = pickle.load(open(feature_file, 'rb'))
    return features['seq']

def get_feature_cov(pdb, features_path, expected_n_channels):
    f = open(features_path + '../fasta/' + pdb + '.fasta', 'r')
    f.readline()
    seq = f.readline().strip()
    L = len(seq)
    X = np.zeros((1, L, L, 441))
    x_ch_first = np.memmap(features_path + pdb + '.cov.21c', dtype=np.float32, mode='r', shape=(1, 441, L, L))
    x = np.rollaxis(x_ch_first[0], 0, 3) # convert to channels_last
    return x

def get_feature(pdb, features_path, expected_n_channels):
    xpre = np.load(features_path + '/pre441/' + pdb + '.pre441.npy')
    l1 = len(xpre[0, :, 0, 0])
    pre = xpre.reshape(l1, l1, 441)
    assert len(pre[0, 0, :]) == 441

    xcov = np.load(features_path + '/cov16bit/' + pdb + '.cov16bit.npy')
    assert len(xcov[0, 0, :]) == 441
	
    xros = np.load(features_path + '/trRos/' + pdb + '.npy')
    assert len(xros[0, 0, :]) == 526
    x_what_we_need_1 = xros[:, :, :84]
    x_what_we_need_2 = xros[:l1, :l1, -1:]
	
    assert len(x_what_we_need_1[0, 0, :85]) == 84
    assert len(x_what_we_need_2[0, 0, -1:]) == 1
     
    x_pdnet1 = get_feature_55(pdb, features_path, 55)
    avgtrROS_512_512_55 = np.load('/nvme2tb/nachammai/FeatureExtraction/AVGpdnet55/250AVG.pdnet55.npy')
    #avgtrROS_512_512_55 = np.zeros((512,512,526),dtype=np.float16)
    avgtrROS_L_L_55 = avgtrROS_512_512_55[:l1,:l1,48:50]
    #x_pdnet2 = avgtrROS_L_L_55
    x_pdnet = np.concatenate((x_pdnet1[:, :, 0:48], x_pdnet1[:, :, 50:55], avgtrROS_L_L_55[:, :, 48:50]), axis=2)    

    x = np.concatenate((xcov, pre, x_pdnet, x_what_we_need_1, x_what_we_need_2), axis = 2)
    assert len(x[0, 0, :]) == expected_n_channels
	
    return x.astype(np.float32)
	
def get_feature_55(pdb, features_path, expected_n_channels):
    features = pickle.load(open(features_path + '/features/' + pdb + '.pkl', 'rb'))
    l = len(features['seq'])
    seq = features['seq']
    # Create X and Y placeholders
    X = np.full((l, l, expected_n_channels), 0.0)
    '''
    # Add secondary structure
    ss = features['ss']
    assert ss.shape == (3, l)
    fi = 0
    for j in range(3):
        a = np.repeat(ss[j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
        '''
    # Add PSSM
    pssm = features['pssm']
    assert pssm.shape == (21, l)
    fi = 6
    for j in range(21):
        a = np.repeat(pssm[j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add SA
    sa = features['sa']
    assert sa.shape == (l, )
    a = np.repeat(sa.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add entrophy
    entropy = features['entropy']
    assert entropy.shape == (l, )
    a = np.repeat(entropy.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add CCMpred
    ccmpred = features['ccmpred']
    assert ccmpred.shape == ((l, l))
    X[:, :, fi] = ccmpred
    fi += 1
    # Add  FreeContact
    freecon = features['freecon']
    assert freecon.shape == ((l, l))
    X[:, :, fi] = freecon
    fi += 1
    # Add potential
    potential = features['potential']
    assert potential.shape == ((l, l))
    X[:, :, fi] = potential
    fi += 1
    assert fi == expected_n_channels
    assert X.max() < 100.0
    assert X.min() > -100.0
    return X

def get_map(pdb, distmap_path, expected_l = -1):
    (ly, seqy, cb_map) = np.load(distmap_path + pdb + '-cb.npy', allow_pickle = True)
    if expected_l > 0:
        assert expected_l == ly
        assert cb_map.shape == ((expected_l, expected_l))
    Y = cb_map
    Y[Y < 1.0] = 1.0
    Y[0, 0] = Y[0, 1]
    Y[ly-1, ly-1] = Y[ly-1, ly-2]
    for q in range(1, ly-1):
        Y[q, q] = ( Y[q, q-1] + Y[q, q+1] ) / 2.0
    assert Y.max() <= 500.0
    assert Y.min() >= 1.0
    return Y

def save_dist_rr(pdb, pred_matrix, feature_file, file_rr):
    sequence = get_sequence(pdb, feature_file)
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    P = np.copy(pred_matrix)
    L = len(P[:])
    for j in range(0, L):
        for k in range(j, L):
            P[j, k] = (P[k, j, 0] + P[j, k, 0]) / 2.0
    for j in range(0, L):
        for k in range(j, L):
            if abs(j - k) < 5:
                continue
            rr.write("%i %i %0.3f %.3f 1\n" %(j+1, k+1, P[j][k], P[j][k]) )
    rr.close()
    print('Written RR ' + file_rr + ' !')

def save_contacts_rr(pdb, pred_matrix, feature_file, file_rr):
    sequence = get_sequence(pdb, feature_file)
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    P = np.copy(pred_matrix)
    L = len(P[:])
    for j in range(0, L):
        for k in range(j, L):
            P[j, k] = (P[k, j, 0] + P[j, k, 0]) / 2.0
    for j in range(0, L):
        for k in range(j, L):
            if abs(j - k) < 5:
                continue
            rr.write("%i %i 0 8 %.5f\n" %(j+1, k+1, (P[j][k])) )
    rr.close()
    print('Written RR ' + file_rr + ' !')

def dist_map_to_bins(Y, bins):
    L = len(Y[:, 0])
    B = np.full((L, L, len(bins)), 0)
    for i in range(L):
        for j in range(L):
            for bin_i, bin_range in bins.items():
                min_max = [float(x) for x in bin_range.split()]
                if Y[i, j] > min_max[0] and Y[i, j] <= min_max[1]:
                    B[i, j, bin_i] = 1
    return B
