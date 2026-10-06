"""Venue adapters: the only code that understands vendor payloads (see ``base``)."""

from cma.adapters.base import (
    MalformedPayloadError,
    SupportsAttribution,
    SupportsResync,
    SupportsSequenceScope,
    VenueAdapter,
)

__all__ = [
    "MalformedPayloadError",
    "SupportsAttribution",
    "SupportsResync",
    "SupportsSequenceScope",
    "VenueAdapter",
]
