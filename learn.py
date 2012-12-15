from utils import *
from sklearn import svm, cross_validation, pipeline, feature_selection, decomposition, preprocessing, base
import pylab as pl
from scipy import stats
import ConfigParser

def collect_stats_anova():
    '''
    Plot graphs on performance of SVM classifier varying feature selection.
    '''
    from math import log10
    
    percentiles = (1, 5, 7, 10, 20, 50, 70, 100)
    x = get_x(logger=logger)
    print x.shape
    y = get_y(logger=logger)
    print y.shape
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