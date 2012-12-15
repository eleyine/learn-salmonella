'''
Various util functions including I/O file handling and filtering routines.
'''

import numpy as np
import pickle
import json
import logging
import ConfigParser
import os

__all__ = ['Indent',
           'Logger',
           'get_sample_labels',
           'get_feature_labels',
           'get_x',
           'get_y',
           'log',
           'filter_by_label']

class Indent:
    def __init__(self):
        self.indent = ''
        self.single_indent = '    '

    def increment(self):
        self.indent += self.single_indent

    def decrement(self):
        self.indent = self.indent[:-len(self.single_indent)]

    def single(self):
        return self.single_indent

    def __str__(self):
        return self.indent

class Logger:
    '''
    Logging utility.
    '''
    import logging

    def __init__(self, verbose_level=1, indent=None):
        # if someone tried to log something before basicConfig is called, 
        # Python creates a default handler that goes to the console and will 
        # ignore further basicConfig calls. Remove the handler if there is one.
        root = logging.getLogger()
        if root.handlers:
            for handler in root.handlers:
                root.removeHandler(handler)

        if indent is None:
            self.indent = Indent()
        else:
            self.indent = indent

        self.verbose = verbose_level

        if verbose_level == 0:
            logging.basicConfig(format='%(message)s', level=logging.WARNING)
        elif verbose_level == 1:
            logging.basicConfig(format='%(message)s', level=logging.INFO)
        else:
            logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    def info(self, message):
        logging.info(self._format(message, str(self.indent) + '>'))

    def debug(self, message):
        self.increment()
        logging.info(self._format(message, str(self.indent) + '%'))
        self.decrement()

    def warn(self, message):
        logging.warning(self._format(message, str(self.indent) + '/!\\'))

    def increment(self):
        self.indent.increment()

    def decrement(self):
        self.indent.decrement()

    def _format(self, message, pre):
        # enforce space at start of message
        if message[0] != ' ':
            message = ' ' + message

        # enfore message is preceded by correct indent and start symbol
        message = pre + message

        # enforce each new line is followed by indent and start symbol
        message = ''.join([char if char != '\n' else char + pre + ' ' for char in message])
        return message

    def clone(self):
        return Logger(verbose_level=self.verbose, indent=self.indent)


def get_sample_labels(organism='human', logger=Logger(verbose_level=1)):
    ''' 
    Get gene labels and return str numpy array of shape [samples]. 
    '''
    logger.info('Retrieving %s gene labels.' % (organism))

    # load numpy object if it exists
    ORG_GENES_NP = config.get('Numpy', organism + '_genes')
    if os.path.isfile(ORG_GENES_NP):
        logger.debug('Found numpy object in %s.' % (ORG_GENES_NP))
        f = open(ORG_GENES_NP, 'r')
        org_genes = np.load(f)
        f.close()
        return org_genes

    # load json file containing all the data
    JSON_FILE = config.get('Data', 'json_file')
    if not os.path.isfile(JSON_FILE):
        logger.debug('JSON file %s does not exist. Proceeding to extract \
        raw data.' % (JSON_FILE))
        extract_data()

    f = open(JSON_FILE, 'r')
    data = json.load(f)
    f.close()
    logger.debug('Done loading JSON data.')

    org_genes = np.array(data[organism+'_genes'])
    logger.debug('Storing numpy object instance in %s.' % (ORG_GENES_NP))
    f = open(ORG_GENES_NP, 'w')
    np.save(f, org_genes)
    f.close()
    return org_genes 

def get_feature_labels(logger=Logger(verbose_level=1)):
    ''' 
    Get TFBS feature labels and return str numpy array of shape [features]. 
    '''
    logger.info('Retrieving feature labels.')

    # load numpy object if it exists
    TFBS_NP = config.get('Numpy', 'tfbs')
    if os.path.isfile(TFBS_NP):
        logger.debug('Found numpy object in %s.' % (TFBS_NP))
        f = open(TFBS_NP, 'r')
        tfbs = np.load(f)
        f.close()
        return tfbs

    # load json file containing all the data
    JSON_FILE = config.get('Data', 'json_file')
    if not os.path.isfile(JSON_FILE):
        logger.debug('JSON file %s does not exist. Proceeding to extract \
        raw data.' % (JSON_FILE))
        extract_data()

    f = open(JSON_FILE, 'r')
    data = json.load(f)
    f.close()
    logger.debug('Done loading JSON data.')

    tfbs = np.array(data['tfbs'])
    logger.debug('Storing numpy object instance in %s.' % (TFBS_NP))
    f = open(TFBS_NP, 'w')
    np.save(f, tfbs)
    f.close()
    return tfbs 

def get_x(organism='human', logger=Logger(verbose_level=1)):
    ''' 
    Parse x (int) features corresponding to given organism and return numpy 
    array of shape [samples, features]. 
    '''
    logger.info('Retrieving %s feature vector.' % (organism))

    # load numpy object if it exists
    ORG_X_NP = config.get('Numpy', organism+'_x')
    if os.path.isfile(ORG_X_NP):
        logger.debug('Found numpy object in %s.' % (ORG_X_NP))
        f = open(ORG_X_NP, 'r')
        org_x = np.load(f)
        f.close()
        return org_x

    # load json file containing all the data
    JSON_FILE = config.get('Data', 'json_file')
    if not os.path.isfile(JSON_FILE):
        logger.debug('JSON file %s does not exist. Proceeding to extract \
        raw data.' % (JSON_FILE))
        extract_data()

    f = open(JSON_FILE, 'r')
    data = json.load(f)
    f.close()
    logger.debug('Done loading JSON data.')

    org_x = np.array(data[organism +'_x'])
    logger.debug('Storing numpy object instance in %s.' % (ORG_X_NP))
    f = open(ORG_X_NP, 'w')
    np.save(f, org_x)
    f.close()
    return org_x

def get_y(organism='human', logger=Logger(verbose_level=1)):
    ''' 
    Parse y (int) class vector corresponding to given organism and return 
    numpy array of shape [samples]. 
    '''
    logger.info('Retrieving %s class vector.' % (organism))

    # load numpy object if it exists
    ORG_Y_NP = config.get('Numpy', organism+'_y')
    if os.path.isfile(ORG_Y_NP):
        logger.debug('Found numpy object in %s.' % (ORG_Y_NP))
        f = open(ORG_Y_NP, 'r')
        org_y = np.load(f)
        f.close()
        return org_y

    # load json file containing all the data
    JSON_FILE = config.get('Data', 'json_file')
    if not os.path.isfile(JSON_FILE):
        logger.debug('JSON file %s does not exist. Proceeding to extract \
        raw data.' % (JSON_FILE))
        extract_data()

    f = open(JSON_FILE, 'r')
    data = json.load(f)
    f.close()
    logger.debug('Done loading JSON data.')

    org_y = np.array(data[organism +'_y'])
    logger.debug('Storing numpy object instance in %s.' % (ORG_Y_NP))
    f = open(ORG_Y_NP, 'w')
    np.save(f, org_y)
    f.close()
    return org_y

def filter_by_label(x, y, label=1):
    '''
    Return subset of genes corresponding to the given label.

    Parameters
    ----------    
        label: integer
        1 if positive (gene is involved in reaction to Salmonella), 
        0 if unknown

    Returns
    --------
        xsubset: 2D nparray
        The subset of genes corresponding to the given label

        ysubset: 1D nparray
        The subset of outputs corresponding to the given label for convenience.
        This essentially returns an array [label] * occurences_of_label.
    '''
    mask = [ obs == label for obs in y]
    mask = np.array(mask, dtype=bool)
    return (x[mask], y[mask])

def store(filename, obj):
    '''
    Pickle an object instance into filename
    '''
    f = open(filename, 'w') 
    pickle.dump(obj,f)
    f.close()

def load(filename):
    '''
    Return object instance pickled in filename
    '''
    f = open(filename, 'r')
    obj = pickle.load(f)
    f.close()
    return obj

def extract_data():
    '''
    Parses specifically formatted textfiles we were provided and stores them 
    in a more easily parsable json file. 

    Side-Effects
    ------------
    Stores a json file with (key, value) pairs: 
        tfbs: 1D str list
        A list of transcription factor binding sites (TFBS) ids corresponding 
        to our feature labels.

        human_genes: 1D str list
        A list of homo sapiens gene identifiers

        human_x: 2D int list
        Feature vector.
        Each row corresponds to the ith human gene in "human_genes".
        Each column corresponds to the number of TFBS with the jth id (in 
        "tfbs") found in the ith gene (in "human_genes").

        mouse_genes: 1D str list
        A list of mus musculus gene identifiers

        mouse_x: 2D int list
        Feature vector.
        Each row corresponds to the ith mouse gene in "mouse_genes".
        Each column corresponds to the number of TFBS with the jth id (in 
        "tfbs") found in the ith gene (in "mouse_genes").   

        human_positives: 1D string list
        List of genes identified as being involved in Salmonella

        human_y: 1D int list
        Class vector for human_x with shape (human_x)
        A list of integers where 1 means the ith gene (in "human_genes") is 
        involved in Salmonella.

        TODO:
        We are missing mouse_y. Perhaps pipe an entrez search via biopython 
        with query: mus musculus[ORGN] AND salmonella[DIS]
        Or on the converse, query each gene name in the "mouse_genes" and 
        see if salmonella is mentioned in the disease genbank category i.e.

        "mouse_genes"[i][ACCN] AND salmonella[DIS] 
    '''
    logger.info('Parsing unformatted dataset into json file.')

    # read human data
    f = open(config.get('Data', 'human_tfbs'),'r')
    human_data = [line.strip().split('\t') for line in f.readlines()]
    f.close()

    # get tfbs
    tfbs = human_data[0]
    logger.debug('Done parsing %i TFBS ids.' % (len(tfbs)))

    # get human_genes and human_x
    human_data = human_data[1:]
    genes_x_tuples = [(gene_info[3], gene_info[4:]) for gene_info in human_data]
    human_genes = [t[0][:t[0].index('_up')] for t in genes_x_tuples]
    human_x = [[int(j) for j in t[1]] for t in genes_x_tuples]
    logger.debug('Done parsing human feature vector with shape (%i, %i).' \
        % (len(human_genes), len(human_x[0])))

    # read human positives
    f = open(config.get('Data', 'human_positives'),'r')
    human_positives = [line.strip('\n') for line in f.readlines()]
    f.close()
    human_y = [1 if gene in human_positives else 0 for gene in human_genes]
    logger.debug('Done parsing human class vector with %i positives.' \
        % (len(human_positives)))

    # read mouse data
    f = open(config.get('Data', 'mouse_tfbs'),'r')
    mouse_data = [line.strip().split('\t') for line in f.readlines()]
    f.close()

    # make sure the TFBS columns are the same
    assert mouse_data[0] == tfbs

    # get mouse_genes and mouse_x
    mouse_data = mouse_data[1:]
    genes_x_tuples = [(gene_info[3], gene_info[4:]) for gene_info in mouse_data]
    mouse_genes = [t[0][:t[0].index('_up')] for t in genes_x_tuples]
    mouse_x = [[int(j) for j in t[1]] for t in genes_x_tuples]
    logger.debug('Done parsing mouse feature vector with shape (%i, %i).' \
        % (len(mouse_genes), len(mouse_x[1])))

    # dump data in json
    data = {'tfbs': tfbs,
            'human_genes': human_genes,
            'human_x': human_x,
            'human_y': human_y,
            'human_positives': human_positives,
            'mouse_genes': mouse_genes,
            'mouse_x': mouse_x 
            }
    JSON_FILE = config.get('Data', 'json_file')
    f = open(JSON_FILE, 'w')
    json.dump(data, f, sort_keys=True)
    f.close()
    logger.debug('Dumped data to %s' % JSON_FILE)

def log(output, fn='log.txt'):
    '''
    Because logging is cooler than printing.
    '''
    f = open(fn, 'a')
    f.write('\n')
    f.write(str(output))
    f.close()

logger = Logger(verbose_level=2)
config = ConfigParser.ConfigParser()
config.read('config.ini')

if __name__ == '__main__':
    # extract_data()
    # print get_y()
    # print get_feature_labels()
    # print get_sample_labels()
    # print get_sample_labels(organism='mouse')
    pass