"""File-backed reference series (ETF / rates / VIX proxies) for ``Venue.REFERENCE``."""

from cma.adapters.reference.series import ReferenceRow, ReferenceSeriesReader, reference_instrument

__all__ = ["ReferenceRow", "ReferenceSeriesReader", "reference_instrument"]
