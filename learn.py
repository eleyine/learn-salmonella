from utils import *
from sklearn import svm, cross_validation, pipeline, feature_selection, decomposition, preprocessing, base
import pylab as pl
import numpy as np
from scipy import stats
import ConfigParser
import math

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
    def __init__(self, percentile, logger=Logger(verbose_level=1)):
        self.percentile = percentile
        self.logger = logger
        self.logger.increment()
        self.da = None
        self.feature_mask = None

    def fit(self, X, y):
        self.logger.info('Fitting model to data.')
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
        self.logger.info('Selecting %ith best features.' % (self.percentile))
        if self.da is None:
            self.fit(X,y)

        assert self.da.shape[0] == X.shape[1]
        # get the indices of the highest discriminating features
        self.logger.debug('Sorting discriminating ability scores')
        da_sorted_indices = self.da.argsort()[::-1] # decreasing order

        # resize X to only include p best features
        self.logger.debug('Resizing X to only include p best features')
        k = int(X.shape[1] * self.percentile / float(100))
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
    def __init__(self, percentile, logger=Logger(verbose_level=1)):
        self.percentile = percentile
        self.logger = logger
        self.logger.increment()

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
        self.logger.info('Fitting model to data.')
        P, P_y = filter_by_label(X, y, label=1)
        U, U_y = filter_by_label(X, y, label=0)
        pr = self._positive_representative_vector(P)
        dist = self._distances(U, pr)
        self.logger.debug('Computed distances of U genes to pr')
        self.logger.debug('Sorted distance vector: \n%s'
                                    % (str(np.sort(np.array(dist)))))
        # get the indices of the most dissimilar genes from pr
        dist_sorted_indices = dist.argsort()[::-1] # decreasing order

        # resize X to only include all the genes in P + the p most dissimilar 
        # genes in U
        k = int(U.shape[0] * self.percentile / float(100))
        k_best_indices = dist_sorted_indices[:k]
        reduced_U = U[k_best_indices]
        reduced_Uy = U_y[k_best_indices]
        Xt = np.concatenate((P, reduced_U))
        Yt = np.concatenate((P_y, reduced_Uy))
        return (Xt, Yt)

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
                feature_percentile,
                sample_percentile,
                C=None,
                positive_weight=None,
                logger=Logger(verbose_level=1),
                score_specificity=True):
        self.feature_percentile = feature_percentile
        self.sample_percentile = sample_percentile
        self.C = C
        self.positive_weight = None #TODO
        self.logger=logger
        self.logger.info('Classifier init')
        self.logger.increment()
        self.feature_selection = PUDI_FeatureSelection(
            percentile=self.feature_percentile, logger=self.logger.clone())
        self.sample_selection = PUDI_SampleSelection(
            percentile=self.sample_percentile, logger=self.logger.clone())
        self.score_specificity = score_specificity
        self.feature_mask = None

    def fit(self, X, y):
        '''
        Fit the SVM model according to the given training data
        '''
        self.logger.info('Fitting model to data')
        X = self.feature_selection.fit_transform(X, y)
        X, y = self.sample_selection.fit_transform(X, y)
        self.feature_mask = self.feature_selection.get_feature_mask()
        if self.C:
            self.clf = svm.SVC(C=self.C).fit(X, y)
        else:
            self.clf = svm.SVC().fit(X, y)            
        return self

    def predict(self, X):
        self.logger.info('Predicting test set')
        X = X[:, self.feature_mask]
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
        if self.score_specificity:
            total_positives = (predicted == 1).sum()
            true_positives = 0
            for i in range(len(y)):
                if y[i] == 1 and predicted[i] == 1:
                    true_positives += 1
            self.logger.debug('%i true positives / %i' 
                        % (true_positives, total_positives))
            if total_positives == 0:
                s = 0
            else:
                s = true_positives / float(total_positives)
        else:
            s = np.mean(predicted == y)
        self.logger.debug('Score: %f' % (s))
        return s


def collect_stats():
    '''
    Plot graphs on performance of SVM classifier varying feature selection.
    '''
    from math import log10
    
    percentiles = (1, 5, 7, 10, 20, 50, 70, 100)
    x = get_x(logger=logger)
    y = get_y(logger=logger)
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
        logger.debug('p = %i' % (p))
        clf.set_params(pudi__feature_percentile=p)
        # Compute cross-validation score using all CPUs
        this_scores = cross_validation.cross_val_score(clf, x, y, n_jobs=1)
        logger.increment()
        logger.debug('Mean: %f' % (this_scores.mean()))
        logger.debug('Std: %f' % (this_scores.std()))
        logger.decrement()
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())
    log(title)
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

if __name__ == '__main__':
    collect_stats()



