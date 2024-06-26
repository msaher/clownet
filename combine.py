"""Combine testing results of the three models to get final accuracy."""

import argparse
import numpy as np
from utils import ArgsObject

def main(d):
    # parser.add_argument('--iframe', type=str, required=True,
    #                     help='iframe score file.')
    # parser.add_argument('--mv', type=str, required=True,
    #                     help='motion vector score file.')
    # parser.add_argument('--res', type=str, required=True,
    #                     help='residual score file.')

    # parser.add_argument('--wi', type=float, default=2.0,
    #                     help='iframe weight.')
    # parser.add_argument('--wm', type=float, default=1.0,
    #                     help='motion vector weight.')
    # parser.add_argument('--wr', type=float, default=1.0,
    #                     help='residual weight.')

    args = ArgsObject(d)

    with np.load(args.iframe) as iframe:
          with np.load(args.res) as residual:
              n = len(iframe['scores'])

              i_score = np.array([score for score in iframe['scores']])
              # mv_score = np.array([score[0][0] for score in mv['scores']])
              res_score = np.array([score for score in residual['scores']])

              i_label = np.array([score for score in iframe['labels']])
              # mv_label = np.array([score[1] for score in mv['scores']])
              res_label = np.array([score for score in residual['labels']])
              assert np.alltrue(i_label == mv_label) and np.alltrue(i_label == res_label)

              combined_score = i_score * args.wi + res_score * args.wr

              accuracy = float(sum(np.argmax(combined_score, axis=1) == i_label)) / n
              print('Accuracy: %f (%d).' % (accuracy, n))

if __name__ == '__main__':
    main()
