# definition of a class to hold the args dictionary to avoid changing too much code
class ArgsObject:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)
