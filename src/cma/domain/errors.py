"""Exception hierarchy. Every platform error derives from :class:`CMAError`."""

from __future__ import annotations


class CMAError(Exception):
    """Base class for platform errors."""


class InvalidProbabilityError(CMAError, ValueError):
    """A price/probability fell outside [0, 1] or was not a finite number."""


class InvalidPriceError(CMAError, ValueError):
    """A price is malformed, negative, or does not conform to the venue tick structure."""


class ImmutableTimestampError(CMAError):
    """Attempt to overwrite a timestamp that has already been recorded."""


class LookaheadError(CMAError):
    """A computation used information stamped after its decision watermark."""


class StaleDataError(CMAError):
    """Required market data is older than the configured maximum age."""


class BookIntegrityError(CMAError):
    """An order book is in an inconsistent state (gap, crossed, negative size...)."""


class MappingNotApprovedError(CMAError):
    """A contract mapping does not have the review status required for the action."""


class LiveTradingDisabledError(CMAError):
    """Live order routing is disabled in v1 and refuses to initialise."""


class ConfigError(CMAError, ValueError):
    """Invalid or unsafe configuration."""


class FinalTestAccessError(CMAError):
    """Training/selection code attempted to read the locked final-test partition."""


class ExpiryMismatchError(CMAError):
    """An option expiry does not match a contract expiry and no explicit rule allows it."""


class NonexistentLocalTimeError(CMAError, ValueError):
    """A local wall-clock time does not exist (spring-forward DST gap)."""


class AmbiguousLocalTimeError(CMAError, ValueError):
    """A local wall-clock time occurs twice (fall-back DST overlap) and no fold was given."""


class ExperimentPublicationError(CMAError):
    """An experiment report is missing mandatory content (latency/cost scenarios, manifest...)."""


class StrategyConfigMismatchError(CMAError):
    """A strategy's model artefact does not match the configuration it is started with."""
