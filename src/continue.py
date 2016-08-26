import numpy as np

import argparse
import lasagne
import featchar
import logging
from dataset import Dset
from exper import setup_logger, Batcher, Reporter, Validator
from lazrnn import RDNN, RDNN_Dummy


def main(args, rnn_param_values):
    # args = get_args()
    setup_logger(args)

    try:
        if args['seed']:
            rng = np.random.RandomState(args['seed'])
            lasagne.random.set_rng(rng)
        dset = Dset(**args)
        feat = featchar.Feat(args['feat'])
        feat.fit(dset, xdsets=[Dset(dname) for dname in args['charset']])

        batcher = Batcher(args['n_batch'], feat)
        reporter = Reporter(dset, feat)

        validator = Validator(dset, batcher, reporter)

        RNN = RDNN_Dummy if args['rnn'] == 'dummy' else RDNN
        rdnn = RNN(feat.NC, feat.NF, args)
        rdnn.set_param_values(rnn_param_values)

        validator.validate(rdnn, args)
    except Exception as e:
        print e
        logging.exception('an error occured.')

if __name__ == '__main__':
    from utils import MODEL_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='model file to load from')
    parser.add_argument('log_file', help='log file and new model file name')
    cargs = vars(parser.parse_args())

    dat = np.load('{}/{}'.format(MODEL_DIR, cargs['model_file']))
    args = dat['argsd'].tolist()
    args['log'] = cargs['log_file']
    main(args, dat['rnn_param_values'])


