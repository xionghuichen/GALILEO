# Created by xionghuichen at 2022/7/25
# Email: chenxh@lamda.nju.edu.cn


# Copyright (c) 2020, Ioana Bica

from __future__ import print_function

import numpy as np
import pickle

from sklearn.model_selection import StratifiedShuffleSplit


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def compute_beta(alpha, optimal_dosage):
    if (optimal_dosage <= 0.001 or optimal_dosage >= 1.0):
        beta = 1.0
    else:
        beta = (alpha - 1.0) / float(optimal_dosage) + (2.0 - alpha)

    return beta




def get_patient_outcome(x, v, treatment, dosage, scaling_parameter=10):
    if (treatment == 0):
        y = float(scaling_parameter) * (np.dot(x, v[0]) + 12.0 * dosage * (dosage - 0.75 * (
                np.dot(x, v[1]) / np.dot(x, v[2]))) ** 2)
    elif (treatment == 1):
        y = float(scaling_parameter) * (np.dot(x, v[0]) + np.sin(
            np.pi * (np.dot(x, v[1]) / np.dot(x, v[2])) * dosage))
    elif (treatment == 2):
        y = float(scaling_parameter) * (np.dot(x, v[0]) + 12.0 * (np.dot(x, v[
            1]) * dosage - np.dot(x, v[2]) * dosage ** 2))
    else:
        raise NotImplementedError
    return y


def get_dataset_splits(dataset):
    dataset_keys = ['x', 't', 'd', 'y', 'y_normalized']

    train_index = dataset['metadata']['train_index']
    val_index = dataset['metadata']['val_index']
    test_index = dataset['metadata']['test_index']

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index]
        dataset_val[key] = dataset[key][val_index]
        dataset_test[key] = dataset[key][test_index]

    dataset_train['metadata'] = dataset['metadata']
    dataset_val['metadata'] = dataset['metadata']
    dataset_test['metadata'] = dataset['metadata']

    return dataset_train, dataset_val, dataset_test


def get_split_indices(num_patients, patients, treatments, validation_fraction, test_fraction):
    num_validation_patients = int(np.floor(num_patients * validation_fraction))
    num_test_patients = int(np.floor(num_patients * test_fraction))

    test_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_test_patients, random_state=0)
    rest_indices, test_indices = next(test_sss.split(patients, treatments))

    val_sss = StratifiedShuffleSplit(n_splits=1, test_size=num_validation_patients, random_state=0)
    train_indices, val_indices = next(val_sss.split(patients[rest_indices], treatments[rest_indices]))

    return train_indices, val_indices, test_indices


class TCGA_Data():
    def __init__(self, args):
        np.random.seed(3)

        self.select_treatment = args['select_treatment']
        self.treatment_selection_bias = args['treatment_selection_bias']
        self.dosage_selection_bias = args['dosage_selection_bias']

        self.validation_fraction = args['validation_fraction']
        self.test_fraction = args['test_fraction']

        self.tcga_data = pickle.load(open('datasets/tcga.p', 'rb'))
        self.patients = self.normalize_data(self.tcga_data['rnaseq'])

        self.scaling_parameter = 10
        self.noise_std = 0.2

        self.num_weights = 3
        self.v = np.zeros(shape=(self.num_weights, self.patients.shape[1]))
        for j in range(self.num_weights):
            self.v[j] = np.random.uniform(0, 10, size=(self.patients.shape[1]))
            self.v[j] = self.v[j] / np.linalg.norm(self.v[j])

        self.dataset = self.generate_dataset()

    def normalize_data(self, patient_features):
        x = (patient_features - np.min(patient_features, axis=0)) / (
                np.max(patient_features, axis=0) - np.min(patient_features, axis=0))

        for i in range(x.shape[0]):
            x[i] = x[i] / np.linalg.norm(x[i])

        return x

    def generate_dataset(self, batch_size=-1):
        tcga_dataset = dict()
        tcga_dataset['x'] = []
        tcga_dataset['y'] = []
        tcga_dataset['t'] = []
        tcga_dataset['d'] = []
        tcga_dataset['metadata'] = dict()
        tcga_dataset['metadata']['v'] = self.v
        tcga_dataset['metadata']['treatment_selection_bias'] = self.treatment_selection_bias
        tcga_dataset['metadata']['dosage_selection_bias'] = self.dosage_selection_bias
        tcga_dataset['metadata']['noise_std'] = self.noise_std
        tcga_dataset['metadata']['scaling_parameter'] = self.scaling_parameter
        if batch_size>0:
            idx = np.random.randint(0, len(self.patients), batch_size)
            np.random.shuffle(idx)
            selected_patients = self.patients[idx]
        else:
            selected_patients = self.patients
        for patient in selected_patients:
            t, dosage, y = self.generate_patient(x=patient)
            tcga_dataset['x'].append(patient)
            tcga_dataset['t'].append(t)
            tcga_dataset['d'].append(dosage)
            tcga_dataset['y'].append(y)

        for key in ['x', 't', 'd', 'y']:
            tcga_dataset[key] = np.array(tcga_dataset[key])

        tcga_dataset['metadata']['y_min'] = np.min(tcga_dataset['y'])
        tcga_dataset['metadata']['y_max'] = np.max(tcga_dataset['y'])

        tcga_dataset['y_normalized'] = (tcga_dataset['y'] - np.min(tcga_dataset['y'])) / (
                np.max(tcga_dataset['y']) - np.min(tcga_dataset['y']))

        train_indices, validation_indices, test_indices = get_split_indices(num_patients=tcga_dataset['x'].shape[0],
                                                                            patients=tcga_dataset['x'],
                                                                            treatments=tcga_dataset['t'],
                                                                            validation_fraction=self.validation_fraction,
                                                                            test_fraction=self.test_fraction)

        tcga_dataset['metadata']['train_index'] = train_indices
        tcga_dataset['metadata']['val_index'] = validation_indices
        tcga_dataset['metadata']['test_index'] = test_indices

        return tcga_dataset

    def generate_patient(self, x, dosage=None):
        outcomes = []
        dosages = []

        # for treatment in range(num_treatments):
        if (self.select_treatment == 0):
            b = 0.75 * np.dot(x, self.v[1]) / (np.dot(x, self.v[2]))
            if dosage is None:
                if (b >= 0.75):
                    optimal_dosage = b / 3.0
                else:
                    optimal_dosage = 1.0
            else:
                optimal_dosage = np.ones(b.shape)
                optimal_dosage[b >= 0.75] = b[b>=0.75]/3

            alpha = self.dosage_selection_bias
            if dosage is None:
                dosage = np.random.beta(alpha, compute_beta(alpha, optimal_dosage))
            else:
                dosage = dosage[..., 0]

            y = get_patient_outcome(x, self.v, self.select_treatment, dosage, self.scaling_parameter)

        elif (self.select_treatment == 1):
            optimal_dosage = np.dot(x, self.v[2]) / (2.0 * np.dot(x, self.v[1]))
            alpha = self.dosage_selection_bias
            if dosage is None:
                dosage = np.random.beta(alpha, compute_beta(alpha, optimal_dosage))
                if (optimal_dosage <= 0.001):
                    dosage = 1 - dosage
            else:
                dosage = dosage[..., 0]
                dosage[optimal_dosage <= 0.001] = 1 - dosage[optimal_dosage <= 0.001]

            y = get_patient_outcome(x, self.v, self.select_treatment, dosage, self.scaling_parameter)

        elif (self.select_treatment == 2):
            optimal_dosage = np.dot(x, self.v[1]) / (2.0 * np.dot(x, self.v[2]))
            alpha = self.dosage_selection_bias
            if dosage is None:
                dosage = np.random.beta(alpha, compute_beta(alpha, optimal_dosage))
                if (optimal_dosage <= 0.001):
                    dosage = 1 - dosage
            else:
                dosage = dosage[..., 0]
                dosage[optimal_dosage <= 0.001] = 1 - dosage[optimal_dosage <= 0.001]
            y = get_patient_outcome(x, self.v, self.select_treatment, dosage, self.scaling_parameter)
        else:
            raise NotImplementedError
        if dosage is None:
            y = y + np.random.normal(0, self.noise_std)
        else:
            y = np.expand_dims(y, axis=-1)
        return self.select_treatment, dosage, y


class TcgaEnv(object):
    def __init__(self, tcga_data):
        self.tcga_data_class = tcga_data
        assert isinstance(self.tcga_data_class, TCGA_Data)
        pass

    def reset(self, batch_size):
        dataset = self.tcga_data_class.generate_dataset(batch_size)
        return dataset['x']

    def complete_env(self, s, a, ps):
        return s

    def part_env(self, s, a):
        return self.tcga_data_class.generate_patient(s, a)[2]

    def step(self, s, a):
        ps = self.part_env(s, a)
        next_s = self.complete_env(s, a, ps)
        return next_s, ps, np.zeros(ps.shape), ps