"""There are a handful of different errors that we can report. This houses all
of them and provides information regarding ways to fix them."""


class InvalidStrategyException(Exception):
    """Thrown when a scoring strategy is invalid"""

    def __init__(self, strategy, msg=None, options=None):
        if msg is None:
            msg = "%s is not a valid strategy for determining the optimal variable. " % strategy
            msg += "\nShould be a callable or a valid string option. "
            if options is not None:
                msg += "Valid options are %r" % options

        super(InvalidStrategyException, self).__init__(msg)
        self.strategy = strategy
        self.options = None


class InvalidDataException(Exception):
    """Thrown when the training or scoring data is not of the right type"""

    def __init__(self, data, msg=None):
        if msg is None:
            msg = "Data is not of the right format"

        super(InvalidDataException, self).__init__(msg)
        self.data = data
