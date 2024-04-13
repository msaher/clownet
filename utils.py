# definition of a class to hold the args dictionary to avoid changing too much code
from typing import Any
from models import Clownet


class ArgsObject:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)

def create_model(args: Any, cuda=True) -> Clownet:
    # TODO: what's this?
    # global best_prec1

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    else:
        raise ValueError('Unknown dataset '+ args.data_name)

    if cuda:
        model = Clownet(num_class, args.num_segments, args.representation).to('cuda')
    else:
        model = Clownet(num_class, args.num_segments, args.representation)

    return model

