import numpy as np
from cobel.analysis.field_classification_helpers import place_like, modulated, empty_fields, vector_cells
import os

N_UNITS = 50


def stats(activations):
    data_place = []
    data_empty = []
    data_modulated = []
    data_vector = []

    fields, ids, pd, v = place_like(activations)
    data_place.append([fields, ids, pd, v])
    ids_all, ids_any = empty_fields(activations)
    data_empty.append([ids_all, ids_any])
    fls, i, pa, va = modulated(activations)
    data_modulated.append([fls, i, pa, va])
    fs, idxs, ps, vs = vector_cells(activations)
    data_vector.append([fs, idxs, ps, vs])

    return data_place, data_empty, data_modulated, data_vector


def classify_modulated_vector(idx_modulated, idx_vector):
    mask = np.in1d(idx_modulated, idx_vector)
    mask2 = np.in1d(idx_vector, idx_modulated)
    id_modulated = idx_modulated[~mask]
    id_vector = idx_modulated[mask]
    return mask, mask2, id_modulated, id_vector


def reclassify(data_modulated, data_vector):
    for m, v in zip(data_modulated, data_vector):
        mask, mask2, ids_mod, ids_vec = classify_modulated_vector(m[1], v[1])
        m[1] = ids_mod
        v[1] = ids_vec
        m[0] = (m[0])[~mask]
        v[0] = (v[0])[mask2]


def calc_other_ids(class_ids, n_units=N_UNITS):
    ''' class ids is a list of the indices of units in the recurrent layer
    that have been classified as different cell types. This is a list of
    5 numpy arrays which are as follows:
        0 - place like
        1 - no activity
        2 - partially active
        3 - direction modulated
        4 - vector like
    This function returns the cell indices that are not classified as any of
    these types as a numpy array
    '''
    all_ids = np.arange(0, n_units)
    classified = np.concatenate([c if c.shape != () else c.reshape(1, ) for c in class_ids])
    other_ids = np.setdiff1d(all_ids, classified)
    return other_ids


def classify_fields(sim_folder, activations_prefix='activations_trial_'):
    folders = os.listdir(sim_folder)
    run_folders = [folder for folder in folders if folder.startswith('run')]
    run_folders.sort()

    data_list = []
    for run_folder in run_folders:
        run = run_folder.lstrip('run_')  # get which run it was
        activation_folder = sim_folder + run_folder + '/activations/'
        activation_files = [f for f in os.listdir(activation_folder) if f.startswith(activations_prefix)]
        activation_files.sort()

        for file in activation_files:
            trial = file.replace(activations_prefix, '').rstrip('.npy')
            data_place, data_empty, data_modulated, data_vector = stats(np.load(activation_folder + file))
            reclassify(data_modulated, data_vector)
            place = data_place[0][1]
            empty_all = data_empty[0][0]
            empty_some = data_empty[0][1]
            hd_modulated = data_modulated[0][1]
            vector = data_vector[0][1]
            class_ids = [place, empty_all, empty_some, hd_modulated,
                         vector]
            other = calc_other_ids(class_ids)
            class_names = ['Place', 'No response', 'View selective', 'HD Modulated',
                           'Vector', 'Other']
            class_ids.append(other)
            for i, class_id in enumerate(class_ids):
                for c in class_id:
                    data_list.append([int(run), int(trial), c, class_names[i]])

    return data_list

