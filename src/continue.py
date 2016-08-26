import numpy as np

import featchar
import logging
from dataset import Dset
from exper import setup_args, Batcher, Reporter, Validator
from lazrnn import RDNN, RDNN_Dummy


def main(args):
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
        logging.exception('an error occured.')

if __name__ == '__main__':
    import sys
    from utils import MODEL_DIR
    model_file_name = sys.argv[1]
    dat = np.load('{}/{}'.format(MODEL_DIR,mode_file_name))
    print dat['argsd'].tolist()
