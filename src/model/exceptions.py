class ZeroPException(BaseException):
    """Raised internally when the probability density is zero everywhere.

    Caught by ``PICIP.run`` to trigger an automatic retry with a broader
    ``predicted_error``.  Users should not need to handle this directly.
    """

    pass


class InconsistentDataError(ValueError):
    """The samples, known phases and weights admit no consistent unknown phase.

    Raised by ``PICIP.run``; the message is written for the person running the
    tool (the LMDS handler reports it as an input error).
    """
