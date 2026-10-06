"""Canonical enumerations shared by every layer."""

from __future__ import annotations

from enum import StrEnum


class Venue(StrEnum):
    KALSHI = "KALSHI"
    POLYMARKET = "POLYMARKET"
    COINBASE = "COINBASE"
    BINANCE = "BINANCE"
    DERIBIT = "DERIBIT"
    REFERENCE = "REFERENCE"  # generic file-backed reference series (ETF/rates/...)
    SYNTHETIC = "SYNTHETIC"  # deterministic generated data (tests, simulation studies)


class InstrumentKind(StrEnum):
    BINARY_CONTRACT = "BINARY_CONTRACT"
    SPOT = "SPOT"
    PERPETUAL = "PERPETUAL"
    OPTION = "OPTION"
    INDEX = "INDEX"
    ETF = "ETF"


class EventType(StrEnum):
    BOOK_SNAPSHOT = "BOOK_SNAPSHOT"
    BOOK_DELTA = "BOOK_DELTA"
    TRADE = "TRADE"
    STATUS = "STATUS"


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class BookSide(StrEnum):
    BID = "BID"
    ASK = "ASK"

    @property
    def opposite(self) -> BookSide:
        return BookSide.ASK if self is BookSide.BID else BookSide.BID


class Outcome(StrEnum):
    YES = "YES"
    NO = "NO"


class DeltaMode(StrEnum):
    ABSOLUTE = "ABSOLUTE"  # quantity is the new total resting size at the price
    INCREMENT = "INCREMENT"  # quantity is a signed change to the resting size


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(StrEnum):
    IOC = "IOC"
    FOK = "FOK"
    GTC = "GTC"
    GTD = "GTD"


class OrderState(StrEnum):
    PENDING_ARRIVAL = "PENDING_ARRIVAL"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_ORDER_STATES

    @property
    def is_working(self) -> bool:
        return self in (OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.CANCEL_PENDING)


_TERMINAL_ORDER_STATES = frozenset(
    {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED, OrderState.EXPIRED}
)


class LiquidityRole(StrEnum):
    MAKER = "MAKER"
    TAKER = "TAKER"


class MappingStatus(StrEnum):
    UNMAPPED = "UNMAPPED"  # no strategy may trade
    DRAFT = "DRAFT"  # proposed; research only
    REVIEWED = "REVIEWED"  # semantics manually verified; eligible for backtest
    APPROVED_PAPER = "APPROVED_PAPER"  # eligible for forward paper execution
    LIVE_ELIGIBLE = "LIVE_ELIGIBLE"  # reserved; requires a separate approval process

    @property
    def rank(self) -> int:
        return _MAPPING_RANK[self]

    def at_least(self, other: MappingStatus) -> bool:
        return self.rank >= other.rank


_MAPPING_RANK = {
    MappingStatus.UNMAPPED: 0,
    MappingStatus.DRAFT: 1,
    MappingStatus.REVIEWED: 2,
    MappingStatus.APPROVED_PAPER: 3,
    MappingStatus.LIVE_ELIGIBLE: 4,
}


class ExecutionMode(StrEnum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class ContractStatus(StrEnum):
    UNOPENED = "UNOPENED"
    OPEN = "OPEN"
    CLOSED = "CLOSED"  # trading halted / closed, awaiting settlement
    SETTLED = "SETTLED"
    VOIDED = "VOIDED"
    UNKNOWN = "UNKNOWN"


class SettlementOutcome(StrEnum):
    YES = "YES"
    NO = "NO"
    VOID = "VOID"  # canceled market; treatment governed by the venue rule / policy
    SCALAR = "SCALAR"  # settles at a fractional YES value (e.g. Polymarket 50/50)


class MarkMethod(StrEnum):
    MID = "MID"
    CONSERVATIVE = "CONSERVATIVE"  # longs at bid, shorts at ask
    LAST = "LAST"
    FAIR = "FAIR"
    SETTLEMENT = "SETTLEMENT"


class QualityFlag(StrEnum):
    SEQUENCE_GAP = "SEQUENCE_GAP"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    CROSSED = "CROSSED"
    LOCKED = "LOCKED"
    EMPTY = "EMPTY"
    STALE = "STALE"
    CLOCK_DRIFT = "CLOCK_DRIFT"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    NEGATIVE_QUANTITY = "NEGATIVE_QUANTITY"
    OFF_TICK = "OFF_TICK"
    AWAITING_SNAPSHOT = "AWAITING_SNAPSHOT"
    DISCONNECTED = "DISCONNECTED"
    TOP_OF_BOOK_ONLY = "TOP_OF_BOOK_ONLY"
    SOURCE_TS_MISSING = "SOURCE_TS_MISSING"


# Flags that make a book unusable for signal generation or simulated matching.
BLOCKING_QUALITY_FLAGS = frozenset(
    {
        QualityFlag.SEQUENCE_GAP,
        QualityFlag.CROSSED,
        QualityFlag.LOCKED,
        QualityFlag.STALE,
        QualityFlag.CHECKSUM_MISMATCH,
        QualityFlag.NEGATIVE_QUANTITY,
        QualityFlag.AWAITING_SNAPSHOT,
        QualityFlag.DISCONNECTED,
    }
)


class Operator(StrEnum):
    """Payoff operator of a binary contract on an underlying observation X."""

    GT = "GT"  # YES iff X > strike
    GE = "GE"  # YES iff X >= strike
    LT = "LT"
    LE = "LE"
    BETWEEN = "BETWEEN"  # YES iff low <= X < high (exact inclusivity in mapping notes)
    UP = "UP"  # YES iff X_end > X_start (up/down markets)
    DOWN = "DOWN"


class Decision(StrEnum):
    REJECT = "REJECT"
    COLLECT_MORE_DATA = "COLLECT_MORE_DATA"
    FORWARD_PAPER_CANDIDATE = "FORWARD_PAPER_CANDIDATE"
    CONTINUE_PAPER = "CONTINUE_PAPER"


class ReasonCode(StrEnum):
    """Why a signal or order was suppressed/rejected. Persisted for observability."""

    MAPPING_NOT_APPROVED = "MAPPING_NOT_APPROVED"
    STALE_DATA = "STALE_DATA"
    BOOK_INVALID = "BOOK_INVALID"
    CLOCK_DRIFT = "CLOCK_DRIFT"
    BELOW_THRESHOLD = "BELOW_THRESHOLD"
    NO_LIQUIDITY = "NO_LIQUIDITY"
    KILL_SWITCH = "KILL_SWITCH"
    DAILY_STOP = "DAILY_STOP"
    CONTRACT_LIMIT = "CONTRACT_LIMIT"
    EVENT_LIMIT = "EVENT_LIMIT"
    PORTFOLIO_LIMIT = "PORTFOLIO_LIMIT"
    INSUFFICIENT_CAPITAL = "INSUFFICIENT_CAPITAL"
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"
    LOOKAHEAD = "LOOKAHEAD"
    CONTRACT_NOT_OPEN = "CONTRACT_NOT_OPEN"
    LIMIT_VIOLATION = "LIMIT_VIOLATION"
    FOK_UNFILLABLE = "FOK_UNFILLABLE"
    LIVE_DISABLED = "LIVE_DISABLED"
