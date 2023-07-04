#!/usr/bin/env python
# -*- coding: utf-8 -*-
from util.convert import transform2blanket


def markov_blanket(train_X, train_y):
    # Find markov blanket of Y.
    #     * Ground truth: [X_3, X_4, X_5, X_6]
    #     * Adjacents: [X_3, X_5, X_6]
    #     * Coparents: [X_4]
    import sys

    sys.path.append("../../..")
    # pylint: disable=no-name-in-module
    # run this example from examples directory
    from featuresAlgorithm.mi_fs.mbfs.pycit import MarkovBlanket

    # set settings
    N_SAMPLES = 8000
    CONFIDENCE_LEVEL = 0.95
    K_KNN = 5
    K_PERM = 10
    SUBSAMPLE_SIZE = None
    N_TRIALS = 500
    N_JOBS = 10

    cit_funcs = {
        'it_args': {
            'statistic_args': {
                'k': K_KNN
            },
            'test_args': {
                'statistic': 'mixed_mi',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS
            }
        },
        'cit_args': {
            'statistic_args': {
                'k': K_KNN
            },
            'test_args': {
                'statistic': 'mixed_cmi',
                'k_perm': K_PERM,
                'subsample_size': SUBSAMPLE_SIZE,
                'n_trials': N_TRIALS,
                'n_jobs': N_JOBS
            }
        }
    }

    newX = transform2blanket(train_X)
    mb = MarkovBlanket(newX, train_y, cit_funcs)
    selected = mb.find_markov_blanket(confidence=CONFIDENCE_LEVEL, verbose=True)
    return selected
