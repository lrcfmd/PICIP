class NoUException(BaseException):
    #raised when the decomposition of a composition does not contain u
    pass
class SmallUException(BaseException):
    #raised when the decomposition of a composition does is under the set threshold)
    pass
class OnlyUException(BaseException):
    #raised when the decomposition of a composition is only u
    pass
class ParsingException(BaseException):
    #raised by input handlers when input is incorrect format
    pass
class ConflictingDataException(BaseException):
    #raised when prob. density is null even when predicted error has
    #been increased to two
    pass
class ZeroPException(BaseException):
    #raised when p is 0 eveywhere
    pass
