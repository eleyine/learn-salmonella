from utils import *
from sklearn import svm, cross_validation, pipeline, feature_selection, decomposition, preprocessing, base
import pylab as pl
from scipy import stats
import ConfigParser
import math

'''
A set of functions to predict genes involved in reaction to Salmonella using
machine learning.

The algorithm is based on the paper from Yang et al.:

"Positive-Unlabeled Learning for Disease Gene Identification" (2012)
'''

class PUDI_FeatureSelection(base.BaseEstimator, base.TransformerMixin):
    '''
    Choose distinguishing features that either frequently occured in the 
    disease gene set P but seldom occured in unlabled gene set U (assuming 
    large porition of unknown genes are still negatives) or frequently
    occurred in U but seldom occured in P.
    '''
    def __init__(self, percentile):
        self.percentile = percentile

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
        positive = filter_by_label(X, y, label=1)
        unknown = filter_by_label(X,y, label=0)
        da = _discriminating_ability_score(positive, unknown)

        # get the indices of the highest discriminating features
        da_sorted_indices = da.argsort[::-1] # decreasing order

        # resize X to only include p best features
        k = int(x.shape[1] * self.percentile / float(100))
        k_best_indices = da_sorted_indices[:k]
        Xt = X[:,k_best_indices]
        return Xt

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
        pos_aff = _affinity_vector(positive)
        unk_aff = _affinity_vector(unknown)

        da = np.zeros(n_tfbs)

        for i in range(n_tfbs):
            # avoid dividing by 0
            if pos_aff[i] == 0 or unk_aff[i] == 0:
                da[i] = 0.000000000001 # lol
            else:
                da[i] = (pos_aff[i] + unk_aff[i]) * \ 
                        math.log( (n_pos / float(pos_aff[i])) + \
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

def collect_stats_anova():
    '''
    Plot graphs on performance of SVM classifier varying feature selection.
    '''
    from math import log10
    
    percentiles = (1, 5, 7, 10, 20, 50, 70, 100)
    x = get_x(logger=logger)
    y = get_y(logger=logger)
    transform = feature_selection.SelectPercentile(feature_selection.f_classif)
    clf = pipeline.Pipeline([('anova', transform),('svc', svm.SVC())])

    logger.info('Performing SVM varying feature selection')
    score_means = list()
    score_stds = list()
    pl.figure()
    title = 'Performance of an SVM classifier varying feature selection'
    fn = config.get('Plots', 'root') + 'feature_selection'
    for p in percentiles:
        logger.debug('p = %i' % (p))
        clf.set_params(anova__percentile=p)
        # Compute cross-validation score using all CPUs
        this_scores = cross_validation.cross_val_score(clf, x, y, n_jobs=1)
        logger.increment()
        logger.debug('Mean: %i' % (this_scores.mean()))
        logger.debug('Std: %i' % (this_scores.std()))
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
    collect_stats_anova()

    positives = [] # 2D [|p|, |tfbs|]
    tfbs_count = [0] * positives[0]
    for positive in positives:
        for i in range(len(positive)):
            tfbs[i] += positive[i]
    pr = [i / float(len(positives)) for i in tfbs]

    negatives = [] # 2D negative samples [|n|, tfbs]
    n_tfbs = len(pr)
    avg_distance = 0
    for negative in negatives:
        difference = sum([pow(pr[j] - negative[j], 2) for j in range(n_tfbs)])
        avg_distance += difference

    avg_distance = avg_distance / float(len(negatives))




