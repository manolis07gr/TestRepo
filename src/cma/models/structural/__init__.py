"""Structural / combinatorial / cross-venue consistency constraints (s.10.3-10.4, H2/H4)."""

from cma.models.structural.complement import detect_complement_mispricing
from cma.models.structural.cross_venue import CrossVenueCosts, detect_cross_venue_arbitrage
from cma.models.structural.exhaustive import ExhaustiveSet, detect_exhaustive_mispricing
from cma.models.structural.nested import detect_nested_violations, is_subset_event
from cma.models.structural.types import (
    ContractQuote,
    Leg,
    OpportunityKind,
    StructuralConfig,
    StructuralInputError,
    StructuralOpportunity,
)

__all__ = [
    "ContractQuote",
    "CrossVenueCosts",
    "ExhaustiveSet",
    "Leg",
    "OpportunityKind",
    "StructuralConfig",
    "StructuralInputError",
    "StructuralOpportunity",
    "detect_complement_mispricing",
    "detect_cross_venue_arbitrage",
    "detect_exhaustive_mispricing",
    "detect_nested_violations",
    "is_subset_event",
]
