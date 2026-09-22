#!/usr/bin/env python3
"""LISTING VENUE — is this security listed on a US exchange, and in USD?

THE RULE. US-listed securities only. A foreign COMPANY listed on a US
exchange is fine: TSM is a Taiwanese company trading as an ADR on the NYSE
in dollars, and it has been a real holding. A security whose LISTING is on
a foreign exchange is not: 3443.TW trades on the Taiwan Stock Exchange in
Taiwan dollars, and every dollar figure the pipeline would compute from its
price is wrong by the exchange rate.

WHY NOT THE SUFFIX. `.TW`, `.T`, `.HK`, `.L`, `.DE`, `.KS` and the rest are
a list that will always miss something, and the list is not what makes a
listing foreign. The gate here is the venue the quote comes from. The
suffix is kept only as a CROSS-CHECK that logs disagreement, because a
disagreement means one of the two is wrong and that is worth seeing.

THE ADR TRAP, stated because it is the way an over-eager filter breaks:
TSM reports exchange NYQ, market us_market and currency USD, but
financialCurrency TWD — the company reports its books in Taiwan dollars.
financialCurrency IS NOT THE TRADING CURRENCY and must never gate anything.
A filter that reads it excludes every ADR in the pool.

UNKNOWN IS EXCLUDED, AND LOUD. A name whose venue cannot be established
does not reach the board: that is the safe direction for a universe
filter. But it is recorded with its reason, because a legitimate US name
dropped for a missing field must be visible rather than silently gone.
"""
from __future__ import annotations

import dataclasses
import re

# Yahoo exchange codes for US venues. Codes, not names: the display name
# ("NasdaqGS", "NYSE") is prose and changes; these are the keys the quote
# carries.
US_EXCHANGE_CODES = {
    "NMS",   # Nasdaq Global Select
    "NGM",   # Nasdaq Global Market
    "NCM",   # Nasdaq Capital Market
    "NAS",   # Nasdaq (generic)
    "NYQ",   # NYSE
    "ASE",   # NYSE American (AMEX)
    "PCX",   # NYSE Arca
    "BTS",   # Cboe BZX
    "PSE",   # NYSE Arca (legacy code)
    "NYS",   # NYSE (legacy code)
}
US_MARKET = "us_market"
USD = "USD"
_SUFFIX = re.compile(r"\.([A-Z]{1,3})$")


@dataclasses.dataclass(frozen=True)
class Venue:
    ticker: str
    exchange: str | None
    market: str | None
    currency: str | None
    source: str              # which field decided it
    us_listed: bool
    reason: str              # why, in words, for the exclusion log
    suffix_disagrees: bool = False


def _suffix_says_foreign(ticker: str) -> bool:
    """The cross-check only. A multi-character alphabetic tail is a venue
    suffix (3443.TW, 6701.T, 000660.KS); a single letter is a US class share
    that _norm already turns into a dash (BRK.B -> BRK-B)."""
    m = _SUFFIX.search(str(ticker).strip().upper())
    return bool(m) and len(m.group(1)) >= 1


def classify(ticker: str, fetch) -> Venue:
    """Resolve one ticker's venue. `fetch` returns a dict with any of
    exchange / market / currency and is injected so this is testable without
    a network and so a pin can hand it a missing field on purpose."""
    tk = str(ticker).strip().upper()
    try:
        raw = fetch(tk) or {}
    except Exception as e:
        return Venue(tk, None, None, None, "fetch_failed", False,
                     f"venue lookup failed ({type(e).__name__}) — excluded "
                     f"because an unknown venue must not reach the board")
    exch = (raw.get("exchange") or "").strip().upper() or None
    market = (raw.get("market") or "").strip().lower() or None
    ccy = (raw.get("currency") or "").strip().upper() or None
    suffix_foreign = _suffix_says_foreign(tk)

    if exch:
        us = exch in US_EXCHANGE_CODES
        source = "exchange"
    elif market:
        us = market == US_MARKET
        source = "market"
    else:
        return Venue(tk, exch, market, ccy, "none", False,
                     "no exchange or market field on the quote — excluded and "
                     "logged; a US name dropped this way is a data failure, "
                     "not a verdict", suffix_foreign)

    if us and ccy and ccy != USD:
        # Both readings recorded, neither silently preferred: the venue says
        # US, the trading currency says otherwise, and a name that does not
        # reconcile with itself does not reach the board.
        return Venue(tk, exch, market, ccy, source, False,
                     f"venue {exch or market} reads US but the quote is in "
                     f"{ccy}, not USD — the row does not reconcile with "
                     f"itself and is excluded on both readings", suffix_foreign)

    if us:
        return Venue(tk, exch, market, ccy, source, True,
                     f"listed on {exch or market} in {ccy or 'USD'}",
                     suffix_foreign)
    return Venue(tk, exch, market, ccy, source, False,
                 f"listed on {exch or market}"
                 + (f" in {ccy}" if ccy else "")
                 + " — not a US exchange", suffix_foreign)


def yf_fetch(tk: str) -> dict:  # pragma: no cover - network path
    """The live resolver. fast_info first because it is one cheap call;
    .info only when fast_info leaves the venue unknown."""
    import yfinance as yf
    t = yf.Ticker(tk)
    out = {}
    try:
        fi = t.fast_info
        out = {"exchange": getattr(fi, "exchange", None),
               "currency": getattr(fi, "currency", None)}
    except Exception:
        out = {}
    if not out.get("exchange"):
        try:
            i = t.info
            out = {"exchange": i.get("exchange"), "market": i.get("market"),
                   "currency": i.get("currency")}
        except Exception:
            pass
    return out


def assert_usd(ticker: str, venue: Venue) -> None:
    """The sizing-path assertion. Any price that becomes a dollar figure —
    share count, exposure, a group cap, heat, the 8% single-order cap —
    passes through here first. 3443.TW at TWD 2,000 is about $62, not
    $2,000: sizing it as dollars is wrong by the exchange rate in whichever
    direction the rate happens to run."""
    if not venue.us_listed:
        raise ValueError(f"{ticker}: {venue.reason}")
    if venue.currency and venue.currency != USD:
        raise ValueError(f"{ticker}: price is quoted in {venue.currency}, "
                         f"not USD — refusing to size off it")


class CachedFetch:
    """Venue lookups cached on disk beside the GICS cache, with the same
    shape of contract: a cache hit costs nothing, a miss costs one quote,
    and when remote lookups are switched off a miss returns UNKNOWN rather
    than a guess. Unknown excludes, and the exclusion is logged."""

    def __init__(self, cache_path: str, allow_remote: bool = True,
                 fetch=None, ttl_days: int = 30):
        self.path, self.allow_remote = cache_path, allow_remote
        self._fetch = fetch or yf_fetch
        self.ttl_days, self.misses, self.remote_calls = ttl_days, 0, 0
        try:
            import json as _json
            with open(cache_path) as f:
                self._cache = _json.load(f)
        except Exception:
            self._cache = {}

    def __call__(self, tk: str) -> dict:
        hit = self._cache.get(tk)
        if isinstance(hit, dict) and hit.get("exchange"):
            return hit
        self.misses += 1
        if not self.allow_remote:
            return {}
        self.remote_calls += 1
        got = self._fetch(tk) or {}
        if got.get("exchange") or got.get("market"):
            self._cache[tk] = got
        return got

    def save(self) -> None:
        import json as _json
        import os as _os
        try:
            _os.makedirs(_os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                _json.dump(self._cache, f, indent=1, sort_keys=True)
            _os.replace(tmp, self.path)
        except Exception:
            pass


def filter_us_listed(tickers, fetch):
    """Split a pool into what may be considered and what may not.

    Returns (kept, excluded_rows, disagreements). Every exclusion carries its
    reason so the rotation artifact can show it: a universe filter that drops
    names silently is how a legitimate US name disappears without anybody
    noticing it went."""
    kept, excluded, disagreements = [], [], []
    for tk in sorted(tickers):
        v = classify(tk, fetch)
        if v.us_listed:
            kept.append(tk)
            if v.suffix_disagrees:
                # The venue field wins; the disagreement is recorded because
                # one of the two readings is wrong and neither is picked here.
                disagreements.append({"ticker": tk, "exchange": v.exchange,
                                      "currency": v.currency,
                                      "note": "ticker suffix reads foreign, "
                                              "venue field reads US — venue "
                                              "field governs, both recorded"})
        else:
            excluded.append({"ticker": tk, "exchange": v.exchange,
                             "market": v.market, "currency": v.currency,
                             "decided_by": v.source, "reason": v.reason})
    return kept, excluded, disagreements
