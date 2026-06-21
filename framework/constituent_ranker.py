#!/usr/bin/env python3
"""
Constituent Ranker — For each qualified theme, ranks the underlying constituent
tickers by the same composite (4w + 12w return) methodology used for themes and
returns the top leaders, with per-ticker risk warnings.

Data-source-agnostic: a `fetcher(ticker, period) -> DataFrame` callable is
injected (same contract as ThemeRanker). An optional `earnings_fetcher(ticker)
-> date | None` callable supplies the next earnings date for the
"earnings_within_7d" warning; when absent, that warning is simply skipped.
"""

import datetime


class ConstituentRanker:
    """Rank the constituents of qualified themes by composite momentum."""

    TOP_N = 3                    # leaders returned per theme
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70          # RSI(14) > 70 -> overbought warning
    NEAR_52W_HIGH_PCT = 0.98     # within 2% of 52-week high
    EARNINGS_TRADING_DAYS = 7    # earnings within N trading days

    def __init__(self, config: dict, fetcher, earnings_fetcher=None):
        """
        Args:
            config: Full framework config dict (reads config["themes"]).
            fetcher: function(ticker, period) -> DataFrame with a "Close" column.
            earnings_fetcher: optional function(ticker) -> datetime.date | None.
        """
        self.themes_cfg = config["themes"]
        self.fetcher = fetcher
        self.earnings_fetcher = earnings_fetcher

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def compute(self, qualified_theme_names: list) -> dict:
        """
        Rank constituents for each qualified theme.

        Args:
            qualified_theme_names: theme names (e.g. ["Semis", "Quantum"]) to rank.

        Returns:
            { theme_name: [ {ticker, current_price, return_4w, return_12w,
                             composite_rank, rsi_14, warnings[]}, ... ] }
            (top TOP_N constituents per theme, best composite first).
        """
        ranking_cfg = self.themes_cfg.get("ranking", {})
        by_name = {t["name"]: t for t in self.themes_cfg.get("watchlist", [])}

        leaders = {}
        for theme_name in qualified_theme_names:
            theme = by_name.get(theme_name)
            if not theme:
                continue
            constituents = theme.get("constituents", []) or []
            if not constituents:
                continue
            leaders[theme_name] = self._rank_theme_constituents(constituents, ranking_cfg)
        return leaders

    # ------------------------------------------------------------------
    # Per-theme ranking
    # ------------------------------------------------------------------
    def _rank_theme_constituents(self, tickers: list, ranking_cfg: dict) -> list:
        """Compute returns for each constituent, rank by composite, return top N."""
        computed = []
        for ticker in tickers:
            data = self._compute_constituent(ticker, ranking_cfg)
            if data is not None:
                computed.append(data)

        if not computed:
            return []

        # --- Composite ranking (rank-average, identical to ThemeRanker) ---
        if len(computed) > 1:
            sorted_4w = sorted(computed, key=lambda x: x["return_4w"], reverse=True)
            for i, c in enumerate(sorted_4w):
                c["rank_4w"] = i + 1
            sorted_12w = sorted(computed, key=lambda x: x["return_12w"], reverse=True)
            for i, c in enumerate(sorted_12w):
                c["rank_12w"] = i + 1
            for c in computed:
                c["composite"] = (c["rank_4w"] + c["rank_12w"]) / 2.0
            computed.sort(key=lambda x: x["composite"])
        else:
            computed[0]["composite"] = 1.0

        top = computed[: self.TOP_N]

        # --- Assemble leader rows (earnings lookup only for the top N) ---
        leaders = []
        for i, c in enumerate(top):
            leaders.append({
                "ticker": c["ticker"],
                "current_price": c["current_price"],
                "return_4w": c["return_4w"],
                "return_12w": c["return_12w"],
                "composite_rank": i + 1,
                "rsi_14": c["rsi_14"],
                "warnings": self._build_warnings(c),
            })
        return leaders

    def _compute_constituent(self, ticker: str, ranking_cfg: dict) -> dict:
        """Fetch 1y data and compute returns, RSI, and 52w-high proximity."""
        try:
            df = self.fetcher(ticker, period="1y")
            lookback_4w = ranking_cfg.get("lookback_4w", 20)
            lookback_12w = ranking_cfg.get("lookback_12w", 60)

            if df is None or len(df) < lookback_12w:
                return None

            close = df["Close"]
            current = float(close.iloc[-1])
            price_4w_ago = float(close.iloc[-lookback_4w]) if len(close) >= lookback_4w else float(close.iloc[0])
            price_12w_ago = float(close.iloc[-lookback_12w]) if len(close) >= lookback_12w else float(close.iloc[0])

            ret_4w = ((current - price_4w_ago) / price_4w_ago) * 100
            ret_12w = ((current - price_12w_ago) / price_12w_ago) * 100

            # 52-week high (prefer the High column; fall back to Close)
            high_series = df["High"] if "High" in df.columns else close
            high_52w = float(high_series.max())
            at_52w_high = high_52w > 0 and current >= high_52w * self.NEAR_52W_HIGH_PCT

            rsi = self._rsi([float(x) for x in list(close)], self.RSI_PERIOD)

            return {
                "ticker": ticker,
                "current_price": round(current, 2),
                "return_4w": round(ret_4w, 2),
                "return_12w": round(ret_12w, 2),
                "rsi_14": round(rsi) if rsi is not None else None,
                "at_52w_high": at_52w_high,
                "rsi_overbought": rsi is not None and rsi > self.RSI_OVERBOUGHT,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------
    def _build_warnings(self, c: dict) -> list:
        """Assemble the warning flag list for one constituent (top N only)."""
        warnings = []
        if self._earnings_within_window(c["ticker"]):
            warnings.append("earnings_within_7d")
        if c.get("at_52w_high"):
            warnings.append("at_52w_high")
        if c.get("rsi_overbought"):
            warnings.append("rsi_overbought")
        return warnings

    def _earnings_within_window(self, ticker: str) -> bool:
        """True if the next earnings date is within EARNINGS_TRADING_DAYS trading days."""
        if self.earnings_fetcher is None:
            return False
        try:
            earnings_date = self.earnings_fetcher(ticker)
        except Exception:
            return False
        if earnings_date is None:
            return False
        # Normalize datetime -> date
        if isinstance(earnings_date, datetime.datetime):
            earnings_date = earnings_date.date()
        if not isinstance(earnings_date, datetime.date):
            return False

        today = datetime.date.today()
        if earnings_date < today:
            return False
        return self._trading_days_between(today, earnings_date) <= self.EARNINGS_TRADING_DAYS

    @staticmethod
    def _trading_days_between(start: datetime.date, end: datetime.date) -> int:
        """Count weekdays (Mon-Fri) strictly after `start` up to and including `end`."""
        days = 0
        d = start
        while d < end:
            d += datetime.timedelta(days=1)
            if d.weekday() < 5:  # 0=Mon .. 4=Fri
                days += 1
        return days

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    @staticmethod
    def _rsi(closes: list, period: int = 14):
        """Wilder-smoothed RSI (matches signal_engine.compute_rsi)."""
        if closes is None or len(closes) < period + 1:
            return None
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
