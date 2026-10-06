"""Contract ontology, reviewed mappings and semantic-equivalence checks (scope s.9)."""

from cma.mapping.equivalence import (
    EquivalenceRelation,
    EquivalenceResult,
    check_equivalence,
    family_mismatches,
)
from cma.mapping.parsers import (
    ParseResult,
    SeriesSemantics,
    load_series_semantics,
    parse_kalshi_contract,
    parse_polymarket_contract,
    propose_mapping,
)
from cma.mapping.registry import (
    ApprovalRecord,
    MappingRecord,
    MappingRegistry,
    ReviewRecord,
)
from cma.mapping.review import (
    DEFAULT_OBSERVATION_TOLERANCE_NS,
    MappingReviewError,
    ReviewChecklist,
    is_machine_actor,
    validate_mapping,
)
from cma.mapping.semantics import (
    canonical_event_family,
    mapping_from_dict,
    mapping_to_dict,
    observation_method_spec,
)

__all__ = [
    "DEFAULT_OBSERVATION_TOLERANCE_NS",
    "ApprovalRecord",
    "EquivalenceRelation",
    "EquivalenceResult",
    "MappingRecord",
    "MappingRegistry",
    "MappingReviewError",
    "ParseResult",
    "ReviewChecklist",
    "ReviewRecord",
    "SeriesSemantics",
    "canonical_event_family",
    "check_equivalence",
    "family_mismatches",
    "is_machine_actor",
    "load_series_semantics",
    "mapping_from_dict",
    "mapping_to_dict",
    "observation_method_spec",
    "parse_kalshi_contract",
    "parse_polymarket_contract",
    "propose_mapping",
    "validate_mapping",
]
