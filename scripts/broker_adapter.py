#!/usr/bin/env python3
"""BROKER ADAPTERS — the only source of holdings, quantities and open orders.

Two implementations and no third. ShadowBroker reads a JSON snapshot the
operator maintains and places nothing: it is Stage 0's whole world, and it
is also what the pins run against, because a pin that needs a network is a
pin that passes by accident of the clock.

SchwabBroker is a STUB ON PURPOSE. Access is still under review, and the
one thing that must not happen is a file that looks ready and silently
places a real order the first time a flag flips. Every method raises until
somebody implements it against the granted API, and the open questions are
written into the docstring rather than guessed at.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class OpenOrder:
    ticker: str
    side: str
    qty: int
    placed_for_session: str

    @property
    def key(self) -> str:
        return f"{self.ticker}|{self.placed_for_session}|{self.side}"


class BrokerAdapter(Protocol):
    name: str

    def positions(self) -> dict[str, int]: ...
    def last_price(self, ticker: str) -> float: ...
    def open_orders(self) -> list[OpenOrder]: ...
    def place(self, ticker: str, side: str, qty: int, order_type: str,
              limit: float | None, session: str) -> str: ...
    def cancel(self, order_id: str) -> None: ...
    def token_expiry(self) -> dt.datetime | None: ...


class ShadowBroker:
    """Reads a snapshot; places nothing, ever. `placed` records what a live
    adapter WOULD have been asked to do, so the pins can assert on it."""

    name = "shadow"

    def __init__(self, snapshot: dict):
        self._s = snapshot
        self.placed: list[dict] = []
        self.cancelled: list[str] = []

    @classmethod
    def from_file(cls, path: str) -> "ShadowBroker":
        with open(path) as f:
            return cls(json.load(f))

    def positions(self) -> dict[str, int]:
        return {k: int(v) for k, v in (self._s.get("positions") or {}).items()}

    def last_price(self, ticker: str) -> float:
        return float((self._s.get("prices") or {}).get(ticker, 0.0))

    def open_orders(self) -> list[OpenOrder]:
        return [OpenOrder(**o) for o in (self._s.get("open_orders") or [])]

    def place(self, ticker, side, qty, order_type, limit, session) -> str:
        self.placed.append(dict(ticker=ticker, side=side, qty=qty,
                                order_type=order_type, limit=limit,
                                session=session))
        return f"shadow-{len(self.placed)}"

    def cancel(self, order_id: str) -> None:
        self.cancelled.append(order_id)

    def token_expiry(self) -> dt.datetime | None:
        v = self._s.get("token_expiry")
        return dt.datetime.fromisoformat(v) if v else None


class SchwabBroker:
    """NOT IMPLEMENTED — access is under review (requested; review expected
    around 2026-09-23, app approval 1-3 days after).

    OPEN ITEMS that cannot be answered without access, and must not be
    guessed:
      1. Can an order be QUEUED THE EVENING BEFORE to execute at the next
         open? The whole overnight-veto design (spec rule 7) rests on it.
         If it cannot, the veto window collapses from ~13 hours to the
         minutes between a 09:00-09:25 ET placement and the open.
      2. Which order types survive an out-of-session submission, and
         whether a limit order placed in the evening rests or is rejected.
      3. What the open-orders endpoint returns for an order placed for a
         future session, which rule 3.2's duplicate check depends on.
      4. Whether a 1-share order is accepted on every name in Stage 1.
    """

    name = "schwab"

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "Schwab adapter is deliberately unimplemented until API access "
            "is granted and the operator enables Stage 1")

    def positions(self):  # pragma: no cover - unreachable by construction
        raise NotImplementedError

    def last_price(self, ticker):  # pragma: no cover
        raise NotImplementedError

    def open_orders(self):  # pragma: no cover
        raise NotImplementedError

    def place(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def cancel(self, order_id):  # pragma: no cover
        raise NotImplementedError

    def token_expiry(self):  # pragma: no cover
        raise NotImplementedError
