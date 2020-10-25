"""Basic definitions for the transforms module."""


class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""
    pass


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""
    pass