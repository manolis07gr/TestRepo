"""Persisted data-quality incidents (scope s.16): affected periods can be excluded or
stress-tested explicitly in later research."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from cma.domain.models import stable_id
from cma.storage.db import Database


@dataclass(frozen=True, slots=True)
class Incident:
    venue: str
    instrument_id: str | None
    kind: str  # e.g. SEQUENCE_GAP, DISCONNECTED, CLOCK_DRIFT, QUARANTINE, STALE
    start_ns: int
    end_ns: int | None
    detail: str

    @property
    def incident_id(self) -> str:
        return stable_id("incident", self.venue, self.instrument_id, self.kind, self.start_ns)

    def overlaps(self, start_ns: int, end_ns: int) -> bool:
        end = self.end_ns if self.end_ns is not None else 2**62
        return self.start_ns < end_ns and start_ns < end


@dataclass
class IncidentLog:
    db: Database | None = None
    incidents: list[Incident] = field(default_factory=list)

    def record(self, incident: Incident, created_at_ns: int) -> None:
        self.incidents.append(incident)
        if self.db is not None:
            self.db.execute(
                "INSERT OR IGNORE INTO data_quality_incidents (incident_id, venue, instrument_id, "
                "kind, start_ns, end_ns, detail, created_at_ns) VALUES (?,?,?,?,?,?,?,?)",
                (
                    incident.incident_id,
                    incident.venue,
                    incident.instrument_id,
                    incident.kind,
                    incident.start_ns,
                    incident.end_ns,
                    incident.detail,
                    created_at_ns,
                ),
            )

    def load(self) -> list[Incident]:
        if self.db is None:
            return list(self.incidents)
        rows = self.db.query(
            "SELECT venue, instrument_id, kind, start_ns, end_ns, detail "
            "FROM data_quality_incidents ORDER BY start_ns"
        )
        return [Incident(**row) for row in rows]

    def affecting(
        self, start_ns: int, end_ns: int, instruments: Iterable[str] = ()
    ) -> list[Incident]:
        wanted = set(instruments)
        return [
            i
            for i in self.load()
            if i.overlaps(start_ns, end_ns)
            and (not wanted or i.instrument_id is None or i.instrument_id in wanted)
        ]
