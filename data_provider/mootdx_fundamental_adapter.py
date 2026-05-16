# -*- coding: utf-8 -*-
"""
mootdx fundamental adapter — 通达信财务报表 fallback.

Uses mootdx.affair to download & parse gpcw (个股财务) binary files from
Tongdaxin servers.  Covers growth, earnings (financial_report + dividend via
xdxr), and institution data; capital_flow and dragon_tiger return not_supported.

This adapter is intentionally defensive: ImportError is caught at
instantiation time so the rest of the system never needs to worry about
mootdx being absent.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from .fundamental_adapter import (
        _safe_float,
        _normalize_report_date,
    )
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fundamental_adapter import (  # type: ignore[no-redef]
        _safe_float,
        _normalize_report_date,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants (from mootdx/financial/columns.py, header='zh')
# ---------------------------------------------------------------------------
_COL_ROE = "净资产收益率"
_COL_ROE_WEIGHTED = "加权净资产收益率(每股指标)"
_COL_REVENUE_YOY = "营业收入增长率(%)"
_COL_NET_PROFIT_YOY = "净利润增长率(%)"
_COL_GROSS_MARGIN = "销售毛利率(%)(非金融类指标)"
_COL_OPERATING_CASH_FLOW = "经营活动产生的现金流量净额"
_COL_NET_PROFIT_PARENT = "归属于母公司所有者的净利润"
_COL_REVENUE = "其中：营业收入"
_COL_REVENUE_ALT = "营业收入"

# Institution / holder columns (from gpcw)
_COL_INST_TOTAL = "机构持股总量(股)"
_COL_TOP10_TRADABLE = "十大流通股东持股数量合计(股)"
_COL_TOP10_TOTAL = "十大股东持股数量合计(股)"
_COL_SHAREHOLDER_COUNT = "股东人数(户)"

_NOT_SUPPORTED: Dict[str, Any] = {"status": "not_supported", "source_chain": [], "errors": []}


def _pick_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    """Return first matching column name from candidates."""
    for n in names:
        if n in df.columns:
            return n
    return None


def _normalize_stock_code(code: str) -> str:
    """Normalize a stock code to 6-digit uppercase string (e.g. '600519')."""
    s = str(code).strip().upper()
    # Strip market suffixes
    for prefix in ("SH", "SZ", "BJ"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    if "." in s:
        s = s.split(".", 1)[0]
    # Pad to 6 digits
    return s.zfill(6)


class MootdxFundamentalAdapter:
    """mootdx-based fallback adapter for fundamental financial data."""

    # Default TDX hosts for xdxr queries (same pool as PytdxFetcher)
    _DEFAULT_TDX_HOSTS = [
        ("119.147.212.81", 7709),  # 深圳
        ("112.74.214.43", 7727),   # 深圳
        ("221.231.141.60", 7709),  # 上海
        ("101.227.73.20", 7709),   # 上海
        ("14.215.128.18", 7709),   # 广州
        ("59.173.18.140", 7709),   # 武汉
        ("180.153.39.51", 7709),   # 杭州
    ]

    def __init__(self, tdx_dir: str = ""):
        self._tdx_dir = tdx_dir
        self._available = False
        self._init_error: Optional[str] = None
        try:
            from mootdx.affair import Affair  # noqa: F401
            self._available = True
            logger.info("[mootdx] adapter initialised, tdx_dir=%s", tdx_dir or "<auto>")
        except ImportError as exc:
            self._init_error = str(exc)
            logger.warning("[mootdx] mootdx not installed, adapter disabled: %s", exc)
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("[mootdx] initialisation failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Public API — mirrors AkshareFundamentalAdapter
    # ------------------------------------------------------------------

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """
        Return normalized fundamental blocks from mootdx gpcw data.

        Only covers growth and earnings.financial_report; other sub-blocks
        are left empty (AkShare may fill them in the composite chain).
        """
        if not self._available:
            return {"status": "not_supported", "growth": {}, "earnings": {},
                    "institution": {}, "source_chain": [], "errors": ["mootdx_unavailable"]}

        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        try:
            df = self._fetch_latest_gpcw(stock_code)
        except Exception as exc:
            logger.debug("[mootdx] gpcw fetch/parse failed for %s: %s", stock_code, exc)
            result["errors"].append(f"gpcw:{type(exc).__name__}")
            return result

        if df is None or df.empty:
            result["errors"].append("gpcw:empty")
            return result

        # df has Chinese column headers; code is in the index
        # gpcw files are sorted by date descending, so iloc[0] is the latest report
        row = df.iloc[0]

        # --- growth ---
        roe = _safe_float(row.get(_pick_col(df, _COL_ROE)))
        revenue_yoy = _safe_float(row.get(_pick_col(df, _COL_REVENUE_YOY)))
        net_profit_yoy = _safe_float(row.get(_pick_col(df, _COL_NET_PROFIT_YOY)))
        gross_margin = _safe_float(row.get(_pick_col(df, _COL_GROSS_MARGIN)))

        result["growth"] = {
            "roe": roe,
            "revenue_yoy": revenue_yoy,
            "net_profit_yoy": net_profit_yoy,
            "gross_margin": gross_margin,
        }

        # --- earnings: financial_report ---
        report_date_raw = row.get("report_date")
        report_date = None
        if report_date_raw is not None:
            # mootdx report_date is typically an integer like 20231231
            try:
                rd = int(report_date_raw)
                report_date = f"{rd // 10000:04d}-{(rd // 100) % 100:02d}-{rd % 100:02d}"
            except (TypeError, ValueError):
                report_date = _normalize_report_date(report_date_raw)

        revenue_col = _pick_col(df, _COL_REVENUE, _COL_REVENUE_ALT)
        revenue = _safe_float(row.get(revenue_col)) if revenue_col else None
        net_profit_parent = _safe_float(row.get(_pick_col(df, _COL_NET_PROFIT_PARENT)))
        operating_cash_flow = _safe_float(row.get(_pick_col(df, _COL_OPERATING_CASH_FLOW)))
        # Use weighted ROE for financial report if available, else plain ROE
        roe_weighted = _safe_float(row.get(_pick_col(df, _COL_ROE_WEIGHTED)))
        roe_for_report = roe_weighted if roe_weighted is not None else roe

        financial_report_payload = {
            "report_date": report_date,
            "revenue": revenue,
            "net_profit_parent": net_profit_parent,
            "operating_cash_flow": operating_cash_flow,
            "roe": roe_for_report,
        }
        if any(v is not None for v in financial_report_payload.values()):
            result["earnings"]["financial_report"] = financial_report_payload

        result["source_chain"].append("mootdx_gpcw")

        # --- institution ---
        inst_total = _safe_float(row.get(_pick_col(df, _COL_INST_TOTAL)))
        top10_tradable = _safe_float(row.get(_pick_col(df, _COL_TOP10_TRADABLE)))
        if inst_total is not None:
            result["institution"]["institution_holding_change"] = inst_total
        if top10_tradable is not None:
            result["institution"]["top10_holder_change"] = top10_tradable

        # --- dividend ---
        try:
            xdxr_df = self._fetch_xdxr(stock_code)
            if xdxr_df is not None and not xdxr_df.empty:
                div_payload = self._build_dividend_payload(xdxr_df)
                if div_payload:
                    result["earnings"]["dividend"] = div_payload
                    result["source_chain"].append("mootdx_xdxr")
        except Exception as exc:
            logger.debug("[mootdx] dividend processing failed for %s: %s", stock_code, exc)
            result["errors"].append(f"xdxr:{type(exc).__name__}")

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"

        logger.info(
            "[mootdx] get_fundamental_bundle(%s): status=%s, growth=%s, earnings_keys=%s, institution_keys=%s",
            stock_code, result["status"],
            bool(result["growth"]),
            list(result["earnings"].keys()),
            list(result["institution"].keys()),
        )
        return result

    def get_capital_flow(self, stock_code: str, top_n: int = 5) -> Dict[str, Any]:
        """mootdx does not support capital flow."""
        return dict(_NOT_SUPPORTED)

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        """mootdx does not support dragon-tiger data."""
        return dict(_NOT_SUPPORTED)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_latest_gpcw(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        Download the latest gpcw file and return the row for *stock_code*.

        Steps:
        1. Call Affair.files() to list available remote files.
        2. Pick the latest gpcw*.zip filename.
        3. Download to a local temp dir (or configured tdx_dir).
        4. Parse with Affair.parse().
        5. Filter by stock code (code is DataFrame index).
        """
        from mootdx.affair import Affair

        # 1. List available files
        files = Affair.files()
        if not files:
            logger.debug("[mootdx] Affair.files() returned empty")
            return None

        # 2. Pick latest gpcw file
        gpcw_files = sorted(
            [f for f in files if f.get("filename", "").startswith("gpcw")],
            key=lambda f: f.get("filename", ""),
        )
        if not gpcw_files:
            logger.debug("[mootdx] no gpcw files found in remote listing")
            return None

        latest = gpcw_files[-1]
        filename = latest["filename"]
        logger.debug("[mootdx] latest gpcw file: %s", filename)

        # 3. Download directory
        if self._tdx_dir:
            downdir = self._tdx_dir
            os.makedirs(downdir, exist_ok=True)
        else:
            downdir = os.path.join(tempfile.gettempdir(), "mootdx_gpcw")
            os.makedirs(downdir, exist_ok=True)

        filepath = Path(downdir) / filename
        if not filepath.exists():
            logger.info("[mootdx] downloading %s → %s", filename, downdir)
            Affair.fetch(downdir=downdir, filename=filename)

        # 4. Parse
        if not filepath.exists():
            logger.warning("[mootdx] file not found after download: %s", filepath)
            return None

        df = Affair.parse(downdir=downdir, filename=filename)
        if df is None or df.empty:
            return None

        # 5. Filter by stock code — code is the DataFrame index
        target = _normalize_stock_code(stock_code)
        # mootdx code format: "600519" padded to 6 digits in index
        candidates = [target, target.lstrip("0")]
        for candidate in candidates:
            if candidate in df.index:
                matched = df.loc[[candidate]]
                # If multiple report dates, take the first (latest from sorted file)
                return matched.head(1)

        # Try string comparison on index
        try:
            idx_str = df.index.astype(str).str.strip()
            for candidate in candidates:
                mask = idx_str == candidate
                if mask.any():
                    return df[mask].head(1)
        except Exception:
            pass

        logger.debug("[mootdx] stock code %s not found in gpcw data (index sample: %s)",
                     stock_code, list(df.index[:5]) if len(df.index) >= 5 else list(df.index))
        return None

    def _fetch_xdxr(self, stock_code: str) -> Optional[pd.DataFrame]:
        """Fetch ex-dividend/ex-rights records via pytdx TCP protocol."""
        try:
            from pytdx.hq import TdxHq_API
        except ImportError:
            logger.debug("[mootdx] pytdx not installed, skipping xdxr")
            return None

        api = TdxHq_API()
        market = 1 if stock_code.startswith(("6", "5")) else 0
        try:
            connected = False
            for host, port in self._DEFAULT_TDX_HOSTS:
                try:
                    if api.connect(host, port, time_out=6):
                        connected = True
                        break
                except Exception:
                    continue
            if not connected:
                logger.debug("[mootdx] xdxr connect failed for %s (all hosts)", stock_code)
                return None
            data = api.get_xdxr_info(market, int(stock_code))
        except Exception as exc:
            logger.debug("[mootdx] xdxr fetch failed for %s: %s", stock_code, exc)
            return None
        finally:
            try:
                api.disconnect()
            except Exception:
                pass

        if not data:
            return None
        df = pd.DataFrame(data)
        if 'category' in df.columns and 'fenhong' in df.columns:
            df = df[(df['category'] == 1) & (df['fenhong'] > 0)]
        return df if not df.empty else None

    def _build_dividend_payload(self, xdxr_df: pd.DataFrame) -> Dict[str, Any]:
        """Build dividend payload from pytdx xdxr data, matching AkShare format."""
        now_date = date.today()
        ttm_start = now_date - timedelta(days=365)

        events: List[Dict[str, Any]] = []
        ttm_events: List[Dict[str, Any]] = []

        # Sort by date descending so newest events come first
        xdxr_df = xdxr_df.sort_values(
            by=['year', 'month', 'day'], ascending=[False, False, False]
        ).reset_index(drop=True)

        for _, row in xdxr_df.iterrows():
            year = int(row.get('year', 0) or 0)
            month = int(row.get('month', 0) or 0)
            day = int(row.get('day', 0) or 0)
            if not (year and month and day):
                continue
            try:
                event_date = date(year, month, day)
            except ValueError:
                continue

            fenhong = _safe_float(row.get('fenhong'))
            if fenhong is None or fenhong <= 0:
                continue

            event = {
                "event_date": event_date.isoformat(),
                "cash_dividend_per_share": round(fenhong, 6),
                "is_pre_tax": True,
            }
            events.append(event)

            if event_date >= ttm_start:
                ttm_events.append(event)

        if not events:
            return {}

        return {
            "events": events[:5],
            "ttm_event_count": len(ttm_events),
            "ttm_cash_dividend_per_share": (
                round(sum(e["cash_dividend_per_share"] for e in ttm_events), 6)
                if ttm_events else None
            ),
            "coverage": "cash_dividend_pre_tax",
            "as_of": now_date.isoformat(),
        }
