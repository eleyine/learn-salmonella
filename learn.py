from utils import *
from sklearn import svm, cross_validation, pipeline, feature_selection, decomposition, preprocessing, base
import pylab as pl
from scipy import stats
import ConfigParser

'''
A set of functions to predict genes involved in reaction to Salmonella using
machine learning.

The algorithm is based on the paper from Yang et al.:

"Positive-Unlabeled Learning for Disease Gene Identification" (2012)
'''

class PUDI_FeatureSelection:
    '''
    Choose distinguishing features that either frequently occured in the 
    disease gene set P but seldom occured in unlabled gene set U (assuming 
    large porition of unknown genes are still negatives) or frequently
    occurred in U but seldom occured in P.
    '''
    def __init__(self):
        pass

    def transform(self):
        pass





    def _affinity_vector(gene_set):
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




