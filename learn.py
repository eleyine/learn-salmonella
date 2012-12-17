from utils import *
from sklearn import svm, cross_validation, pipeline, feature_selection, decomposition, preprocessing, base
import pylab as pl
import numpy as np
import ConfigParser
import math
from external import custom_cross_val_score
from sklearn.base import is_classifier, clone
import json

'''
A set of functions to predict genes involved in reaction to Salmonella using
machine learning.

The algorithm is based on the paper from Yang et al.:

"Positive-Unlabeled Learning for Disease Gene Identification" (2012)
'''

__all__ = ['PUDI_FeatureSelection',
           'PUDI_SampleSelection',
           'PUDI_Classifier']

class PUDI_FeatureSelection(base.BaseEstimator, base.TransformerMixin):
    '''
    Choose distinguishing features that either frequently occured in the 
    disease gene set P but seldom occured in unlabled gene set U (assuming 
    large porition of unknown genes are still negatives) or frequently
    occurred in U but seldom occured in P.
    '''
    def __init__(self,
                percentile, 
                logger=Logger(verbose_level=1), 
                tracker=StatsTracker()):
        self.percentile = percentile
        self.logger = logger
        self.logger.increment()
        self.da = None
        self.feature_mask = None
        self.tracker = tracker

    def fit(self, X, y):
        self.logger.info('Fitting PUDI feature selection model to data.')
        positive, _ = filter_by_label(X, y, label=1)
        unknown, _ = filter_by_label(X, y, label=0)
        self.logger.debug('Computing discriminating ability scores')
        self.da = self._discriminating_ability_score(positive, unknown)
        return self

    def transform(self, X, y):
        '''
        Fit model to data and subsequently transform the data

        Parameters
        ----------
        X : numpy array, shape = [n_genes, n_features]
            Training set.

        y : numpy array of shape [n_genes]
            Target values.

        Returns
        -------
        Xt : numpy array, shape = [n_genes, reduced_n_features]
             The training set with reduced features.
        '''
        k = int(X.shape[1] * self.percentile / float(100))
        if k == 0:
            k = 1
        self.logger.info('Selecting %ith best features (%i features).' 
            % (self.percentile, k))
        if self.da is None:
            self.fit(X,y)

        assert self.da.shape[0] == X.shape[1]
        # get the indices of the highest discriminating features
        self.logger.debug('Sorting discriminating ability scores')
        da_sorted_indices = self.da.argsort()[::-1] # decreasing order

        # resize X to only include p best features
        self.logger.debug('Resizing X to only include p best features')

        k_best_indices = da_sorted_indices[:k]
        self.feature_mask = k_best_indices
        Xt = X[:,k_best_indices]
        return Xt

    def fit_transform(self, X, y):
        return self.transform(X, y)

    def get_feature_mask(self):
        return self.feature_mask

    def _discriminating_ability_score(self, positive, unknown):
        '''
        Compute the discriminating ability of each TFBS.

        Parameters
        ----------
            positive: 2D nparray, shape = (n_pos, n_tfbs)
            The positive dataset in which every gene is involved in the
            reaction to Salmonella.

            unknown: 2D nparray, shape = (n_unk, n_tfbs)
            The negative dataset in which genes are not currently known to 
            be involved in the reaction to Salmonella.

        Returns
        -------
            da: 1D nparray, shape = (n_tfbs)
        '''
        assert positive.shape[1] == unknown.shape[1]
        n_pos = positive.shape[0]
        n_unk = unknown.shape[0]
        n_tfbs = positive.shape[1]
        pos_aff = self._affinity_vector(positive)
        unk_aff = self._affinity_vector(unknown)

        da = np.zeros(n_tfbs)

        for i in range(n_tfbs):
            # avoid dividing by 0
            if pos_aff[i] == 0:
                da[i] = 0 # lol
            else:
                if unk_aff[i] == 0:
                    unk_aff[i] = 1 # lolol
                da[i] = (pos_aff[i] + unk_aff[i] ) * math.log( 
                                  (n_pos / float(pos_aff[i])) + \
                                  (n_unk / float(unk_aff[i])) )
        return da

    def _affinity_vector(self, gene_set):
        '''
        Compute the affinity count in the gene set.

        Parameters
        ----------
            gene_set: 2D nparray, shape = (n_genes, n_tfbs)
            A gene set matching each gene to a TFBS feature vector.

        Returns
        ----------
            aff: 1D nparray, shape = (n_tfbs)
            A vector containing the number of times a TFBS was seen
            for all the genes. Example:

        Example
        --------
            gene_set for 3 TFBS and 4 genes:
                [[1,11,0],
                 [0,23,0],
                 [1,0,3],
                 [1,0,0]]

            returns:
                [3,2,1]
        '''
        n_tfbs = gene_set.shape[1]
        n_genes = gene_set.shape[0] 
        aff = np.zeros(n_tfbs)

        for i in range(n_tfbs):
            aff[i] = sum([1 if n > 0 else 0 for n in gene_set[:, i]])
        return aff

class PUDI_SampleSelection(base.BaseEstimator, base.TransformerMixin):
    '''
    Given that we do not have any negative genes, the first step is to 
    extract a set of reliable negative genes RN from U by computing the
    dissimilarities of the unlabeled genes.
    '''
    def __init__(self,
                percentile, 
                logger=Logger(verbose_level=1), 
                tracker=StatsTracker()):
        self.percentile = percentile
        self.logger = logger
        self.logger.increment()
        self.tracker = tracker

    def fit_transform(self, X, y):
        '''
        Fit model to data and subsequently transform the data

        Parameters
        ----------
        X : numpy array, shape = [n_genes, n_features]
            Training set.

        y : numpy array of shape [n_genes]
            Target values.

        Returns
        -------
        Xt : numpy array, shape = [n_genes, reduced_n_features]
             The training set with reduced features.
        '''
        self.logger.info('Selecting %ith most dissimilar U genes.' % (self.percentile))
        self.logger.info('Fitting PUDI sample selection model to data.')

        # retrieve data
        assert X.shape[0] == y.shape[0]
        P, P_y = filter_by_label(X, y, label=1)
        U, U_y = filter_by_label(X, y, label=0)
        labels = get_sample_labels()
        P_labels, _ = filter_by_label(labels, y, label=1)
        U_labels, _ = filter_by_label(labels, y, label=0)

        # obtain positive representative
        pr = self._positive_representative_vector(P)
        self.tracker.add('pr', list(pr))

        # compute euclidean distance from positive representative vector
        # for each gene
        dist = self._distances(U, pr)
        # self.tracker.add('dist', list(dist))
        self.logger.debug('Computed distances of U genes to pr')
        sorted_dist = np.sort(np.array(dist))
        self.logger.debug('Sorted distance vector: \n%s'
                                    % (str(sorted_dist)))
        self.tracker.add('sorted_dist', repr(sorted_dist))

        # get the indices of the most dissimilar genes from pr
        dist_sorted_indices = dist.argsort()[::-1] # decreasing order

        # resize X to only include all the genes in P + the p most dissimilar 
        # genes in U
        k = int(U.shape[0] * self.percentile / float(100))
        k_best_indices = dist_sorted_indices[:k]
        reduced_U = U[k_best_indices]
        reduced_Uy = U_y[k_best_indices]
        reduced_Ulabels = U_labels[k_best_indices]
        Xt = np.concatenate((P, reduced_U))
        Yt = np.concatenate((P_y, reduced_Uy))
        labels = np.concatenate((P_labels, reduced_Ulabels))
        return (Xt, Yt, labels)

    def _positive_representative_vector(self, P):
        '''
        Build a "positive representative vector" (pr) by summing up the 
        genes in P and normalizing it.

        Parameters
        ----------
            P: 2D nparray, shape = (n_pos, n_tfbs)
            The positive dataset in which every gene is involved in the
            reaction to Salmonella.

        Returns
        -------
            pr: 1D nparray, shape = (n_pos)
            The positive representative vector.
        '''
        n_pos = P.shape[0]
        return P.sum(axis=0) / float(n_pos)

    def _distances(self, U, pr):
        '''
        Compute the distance of each gene g_i in U from pr using the Euclidean 
        distance

        Parameters
        ----------
            U: 2D nparray, shape = (n_unk, n_tfbs)
            The unknown dataset in which every gene is not currently known to 
            be involved in the reaction to Salmonella.

            pr: 1D nparray, shape = (n_pos)
            The positive representative vector.

        Returns
        -------
            dist: 1D nparray, shape = (n_unk)
            The average Eucledian distance from the positive representative 
            vector for all genes in U.
        '''
        n_tfbs = pr.shape[0]
        n_unk = U.shape[0]
        dist = np.zeros(n_unk)
        for i in range(n_unk):
            u = U[i]
            dist[i] = np.linalg.norm(pr-u)
        return dist

class PUDI_Classifier(base.BaseEstimator, base.ClassifierMixin):

    # TODO add default values once you know what works well
    def __init__(self,
                # feature_percentile,
                # sample_percentile,
                C=None,
                positive_weight=None,
                logger=Logger(verbose_level=1),
                score_specificity=True):
        # self.feature_percentile = feature_percentile
        # self.sample_percentile = sample_percentile
        self.C = C
        self.positive_weight = None #TODO
        self.logger=logger
        self.logger.increment()
        # self.feature_selection = PUDI_FeatureSelection(
        #     percentile=self.feature_percentile, logger=self.logger.clone())
        # self.sample_selection = PUDI_SampleSelection(
        #     percentile=self.sample_percentile, logger=self.logger.clone())
        self.score_specificity = score_specificity
        self.feature_mask = None

    def fit(self, X, y):
        '''
        Fit the SVM model according to the given training data
        '''
        self.logger.info('Fitting PUDI classifier model to data')
        # X = self.feature_selection.fit_transform(X, y)
        # X, y = self.sample_selection.fit_transform(X, y)
        # self.feature_mask = self.feature_selection.get_feature_mask()
        if self.C:
            self.clf = svm.SVC(C=self.C).fit(X, y)
        else:
            self.clf = svm.SVC().fit(X, y)            
        return self

    def predict(self, X):
        self.logger.info('Predicting test set')
        # X = X[:, self.feature_mask]
        return self.clf.predict(X)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        z : float

        """
        predicted = self.predict(X)
        # if self.score_specificity:
        #     total_positives = (predicted == 1).sum()
        #     true_positives = 0
        #     for i in range(len(y)):
        #         if y[i] == 1 and predicted[i] == 1:
        #             true_positives += 1
        #     self.logger.debug('%i true positives / %i' 
        #                 % (true_positives, total_positives))
        #     if total_positives == 0:
        #         s = 0
        #     else:
        #         s = true_positives / float(total_positives)
        # else:
        s = np.mean(predicted == y)
        self.logger.debug('Score: %f' % (s))
        return s

def classifier_routine(X, y,
                       feature_percentiles=[1,2],
                       sample_percentiles=[20,50],
                       C=None,
                       tracker=StatsTracker(),
                       logger=Logger(verbose_level=1),
                       k=3,
                       ):
    # configure feature selection
    pudi_feature_select = PUDI_FeatureSelection(
                            percentile=feature_percentiles[0],
                            tracker=tracker,
                            logger=logger.clone())
    pudi_feature_select.fit(X, y)

    for feature_p in feature_percentiles:
        logger.info('Feature selection with p = %f' % (feature_p))
        pudi_feature_select.set_params(percentile=feature_p)
        Xft = pudi_feature_select.fit_transform(X, y)

        # configure sample selection
        for sample_p in sample_percentiles:
            title = 'PUDI__feature_p'+str(feature_p) + \
                '__sample_p' + str(sample_p)
            tracker.new(title=title)
            pudi_feature_select.set_params(tracker=tracker)

            logger.info('Sample selection with p = %f' % (sample_p))
            pudi_sample_select = PUDI_SampleSelection(
                    percentile=sample_p,
                    tracker=tracker,
                    logger=logger.clone())

            Xt, yt, all_labels = pudi_sample_select.fit_transform(Xft, y)

            # configure classifier
            pudi_classifier = PUDI_Classifier(
                            C=C,
                            logger=logger.clone())
            TP, OP, FP, TN, ON, FN = [], [], [], [], [], []
            # start cross-validation

            # partition Xt in k groups 
            indices = np.random.permutation(Xt.shape[0])
            group_masks = np.array_split(indices, k)

            for i in range(k):
                logger.debug('cross-validation k = %i' % (i))
                test_mask = group_masks[i]
                train_mask = np.concatenate([group_masks[j] for j in range(k) if j != i])
                x_test = Xt[test_mask]
                x_train = Xt[train_mask]
                y_test = yt[test_mask]
                y_train = yt[train_mask]
                labels = all_labels[test_mask]
                clf = pudi_classifier.fit(x_train, y_train)
                predicted = clf.predict(x_test)

                TP.extend(get_true_positives(predicted, y_test, labels))
                OP.extend(get_original_positives(predicted, y_test, labels))
                FP.extend(get_false_positives(predicted, y_test, labels))
                TN.extend(get_true_negatives(predicted, y_test, labels))
                ON.extend(get_original_negatives(predicted, y_test, labels))
                FN.extend(get_false_negatives(predicted, y_test, labels))

            tracker.add('num_true_positives', len(TP))
            tracker.add('true_positives', list(TP))
            tracker.add('num_original_positives', len(OP))
            tracker.add('original_positives', list(OP))
            tracker.add('num_false_positives', len(FP))
            tracker.add('false_positives', list(FP))
            tracker.add('num_true_negatives', len(TN))
            # tracker.add('true_negatives', list(TN))
            tracker.add('num_original_negatives', len(ON))
            # tracker.add('original_negatives', list(ON))
            tracker.add('num_false_negatives', len(FN))
            tracker.add('false_negatives', list(FN))
            try:
                tracker.add('sensitivity', len(TP) / float(len(TP) + len(FP)))
            except ZeroDivisionError:
                tracker.add('sensitivity', 'N/A')
            tracker.add('specificity', len(TN) / float(len(TN) + len(FN)))
            tracker.add('ratio_correctly_positive', len(TP) / float(len(OP)))
            tracker.add('ratio_correctly_negative', len(TN) / float(len(ON)))
            tracker.store()


    
def get_true_positives(predicted, y, labels):
    n_samples = len(labels)
    return [labels[i] for i in range(n_samples) if predicted[i] == 1 and y[i] == 1]

def get_original_positives(predicted, y, labels):
    n_samples = len(labels)
    return [labels[i] for i in range(n_samples) if y[i] == 1]

def get_false_positives(predicted, y, labels):
    n_samples = len(labels)
    return [labels[i] for i in range(n_samples) if predicted[i] == 1 and y[i] == 0 ]

def get_true_negatives(predicted, y, labels):
    n_samples = len(labels)
    return [labels[i] for i in range(n_samples)  if predicted[i] == 0 and y[i] == 0 ]

def get_original_negatives(predicted, y, labels):
    n_samples = len(labels)
    return [labels[i] for i in range(n_samples) if y[i] == 0 ]

def get_false_negatives(predicted, y, labels):
    n_samples = len(labels)
    return [labels[i] for i in range(n_samples) if predicted[i] == 0 and y[i] == 1]

def collect_stats():
    '''
    Plot graphs on performance of SVM classifier varying feature selection.
    '''
    from math import log10
    
    percentiles = (0.05, 0.1, 0.5, 1, 2, 3, 5, 7, 10, 20, 30, 50)
    x = get_x(logger=logger)
    y = get_y(logger=logger)
    # pudi_feature_select = PUDI_FeatureSelection(
    #                         percentile=1,
    #                         logger=logger.clone())
    # pudi_sample_select = PUDI_SampleSelection(
    #                         percentile=50,
    #                         logger=logger.clone())
    # pudi_classifier = PUDI_Classifier(
    #             feature_percentile=30,
    #             sample_percentile=50,
    #             logger=logger.clone())

    # clf = pipeline.Pipeline([('feature', pudi_feature_select),
    #     ('sample', pudi_sample_select),
    #     ('classifier', pudi_classifier)])

    transform = PUDI_Classifier(
                feature_percentile=30,
                sample_percentile=50,
                logger=logger.clone())
    clf = pipeline.Pipeline([('pudi', transform)])


    logger.info('Performing SVM varying feature selection')
    score_means = list()
    score_stds = list()
    pl.figure()
    title = 'Performance of an SVM classifier varying feature selection'
    fn = config.get('Plots', 'root') + 'feature_selection'
    for p in percentiles:
        logger.debug('p = %f' % (p))
        clf.set_params(pudi__feature_percentile=p)
        # Compute cross-validation score using all CPUs
        this_scores = custom_cross_val_score(clf, x, y, n_jobs=1)
        logger.increment()
        logger.debug('Mean: %f' % (this_scores.mean()))
        logger.debug('Std: %f' % (this_scores.std()))
        logger.decrement()
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())
    log(title)
    log('Percentiles:')
    log(percentiles)
    log('Score means:')
    log(score_means)
    log('Score standard deviations:')
    log(score_stds)
    log('\n')
    pl.errorbar(percentiles, score_means, np.array(score_stds))
    pl.title(title)
    pl.xlabel('Feature selection')
    pl.ylabel('Prediction rate')
    pl.axis('tight')
    pl.savefig(fn)

logger = Logger(verbose_level=2)
config = ConfigParser.ConfigParser()
config.read('config.ini')
tracker = StatsTracker()

def plot_stats():
    from collections import OrderedDict
    f = open(config.get('Stats', 'tracker'), 'r')
    d = json.load(f)
    d = OrderedDict(sorted(d.iteritems()))
    f.close()
    stats = {}
    for k in d.keys():
        tokens = k.split('__')
        feature_p = float(tokens[1][9:])
        sample_p = float(tokens[2][8:])
        if sample_p not in stats.keys():
            stats[sample_p] = {
                'feature_p': [],
                'sensitivity': [],
                'specificity': [],
                'correctly_positive': [],
                'correctly_negative': []
            }
        stats[sample_p]['feature_p'].append(feature_p)
        if d[k]['sensitivity'] == 'N/A':
            d[k]['sensitivity'] = 0
        stats[sample_p]['sensitivity'].append(d[k]['sensitivity'])
        stats[sample_p]['specificity'].append(d[k]['specificity'])
        stats[sample_p]['correctly_positive'].append(d[k]['ratio_correctly_positive'])
        stats[sample_p]['correctly_negative'].append(d[k]['ratio_correctly_negative'])
    for sample_size in stats.keys():
        pl.figure(figsize=(12, 6))
        title = 'Performance of SVM classifier varying feature selection with a sample ' + \
                'selection of ' + str(sample_size) + '%'
        pl.title(title)
        pl.rcParams["axes.titlesize"] = 10
        feature_p = stats[sample_size]['feature_p']
        plot(feature_p, stats[sample_size]['sensitivity'], style='go-', label='sensitivity = TP / (TP + FP)')
        plot(feature_p, stats[sample_size]['specificity'], style='ro--', label='specificity = TN / (TN + FN)')
        plot(feature_p, stats[sample_size]['correctly_positive'], style='bo:', label='original positives predicted correctly')
        plot(feature_p, stats[sample_size]['correctly_negative'], style='mo-', label='original negatives predicted correctly')
        
        pl.xlabel('Feature selection')
        pl.legend(loc=8,prop={'size':10}, fancybox=True)
        pl.axis((
            -0.1,
            100,
            -0.05,
            1.05))
        pl.grid(True)
        pl.xscale('log')
        pl.axhline(-0.05, linewidth=2, color="black")        # inc. width of x-axis and color it green
        fn = config.get('Plots', 'root') + 'svm__sample_size_' + str(sample_size) + '.png'
        pl.savefig(fn)
        pl.clf()

def plot(X,Y, x_max=None,y_min=None, style='go-', label='mean specificity/sensitivity'):
    from random import choice

    # sort both arrays according to x
    X = np.array(X)
    Y = np.array(Y)

    Y = np.take(Y,X.argsort())
    X.sort()

    pl.subplots_adjust(bottom = 0.1)
    pl.plot(X, Y, style, label=label, linewidth=2) 

    x_min = float(X.min())-float(X.max())*0.02
    y_min = float(Y.min())-float(Y.max())*0.02
    x_max = float(X.max())+float(X.max())*0.02
    y_max = float(Y.max())+float(Y.max())*0.02



if __name__ == '__main__':
    x = get_x(logger=logger)
    y = get_y(logger=logger)
    feature_percentiles = [0.1, 0.5, 1, 3, 5, 10, 20, 30, 40, 50, 75, 100]
    sample_percentiles = [45,55]
    # mask = range(100)
    classifier_routine(x,y,
        feature_percentiles=feature_percentiles, 
        sample_percentiles=sample_percentiles,
        logger=logger)
    plot_stats()



