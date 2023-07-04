#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lin D, Tang X. Conditional infomax learning: An integrated framework for feature extraction and fusion[C]//Computer Vision–ECCV 2006: 9th European Conference on Computer Vision, Graz, Austria, May 7-13, 2006. Proceedings, Part I 9. Springer Berlin Heidelberg, 2006: 68-82.
"""
import numpy as np

from featuresAlgorithm.base import MIFSBase
from util.metrics.mutualInfor import mutual_info, conditional_mi


class CIFE(MIFSBase):

    def __init__(self, F, C, m=3):
        super(CIFE, self).__init__(F, C, m)
        self.gamma = 1
        self.beta = 1

    def feature_selection(self):
        """
           This function implements the basic scoring criteria for linear combination of shannon information term.
           The scoring criteria is calculated based on the formula j_cmi=I(f;y)-beta*sum_j(I(fj;f))+gamma*sum(I(fj;f|y))

           Input
           -----
           self.F: {numpy array}, shape (n_samples, n_features)
               input data, guaranteed to be a discrete data matrix
           y: {numpy array}, shape (n_samples,)
               input class labels
           kwargs: {dictionary}
               Parameters for different feature selection algorithms.
               beta: {float}
                   beta is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
               gamma: {float}
                   gamma is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
               function_name: {string}
                   name of the feature selection function
               n_selected_features: {int}
                   number of features to select

           Output
           ------
           F: {numpy array}, shape: (n_features,)
               index of selected features, F[0] is the most important feature
           J_CMI: {numpy array}, shape: (n_features,)
               corresponding objective function value of selected features
           MIfy: {numpy array}, shape: (n_features,)
               corresponding mutual information between selected features and response

           Reference
           ---------
           Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
           """

        n_samples, n_features = self.F.shape
        # Objective function value for selected features
        J_CMI = []
        # Mutual information between feature and response
        MIfy = []

        # select the feature whose j_cmi is the largest
        # t1 stores I(f;y) for each feature f
        t1 = self.get_fc_score_list("mi")
        # t2 stores sum_j(I(fj;f)) for each feature f
        t2 = np.zeros(n_features)
        # t3 stores sum_j(I(fj;f|y)) for each feature f
        t3 = np.zeros(n_features)

        # make sure that j_cmi is positive at the very beginning
        j_cmi = 1

        while True:
            if len(self.selected_feature) == 0:
                # select the feature whose mutual information is the largest
                idx = np.argmax(t1)
                self.selected_feature.append(idx)
                J_CMI.append(t1[idx])
                MIfy.append(t1[idx])
                f_select = self.F[:, idx]

            if len(self.selected_feature) == self.m:
                break

            # if j_cmi < 0:
            #     break

            # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
            j_cmi = -1E30
            for i in range(n_features):
                if i not in self.selected_feature:
                    f = self.F[:, i]
                    t2[i] += mutual_info(f_select, f)
                    t3[i] += conditional_mi([f_select, f], self.C)
                    # calculate j_cmi for feature i (not in F)
                    t = t1[i] - self.beta * t2[i] + self.gamma * t3[i]
                    # record the largest j_cmi and the corresponding feature index
                    if t > j_cmi:
                        j_cmi = t
                        idx = i
            self.selected_feature.append(idx)
            self.print_sf()
            J_CMI.append(j_cmi)
            MIfy.append(t1[idx])
            f_select = self.F[:, idx]

        return list(self.selected_feature)


if __name__ == '__main__':
    pass
