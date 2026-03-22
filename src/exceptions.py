class ZeroPException(BaseException):
    """Raised internally when the probability density is zero everywhere.

    Caught by ``PICIP.run`` to trigger an automatic retry with a broader
    ``predicted_error``.  Users should not need to handle this directly.
    """

    pass
