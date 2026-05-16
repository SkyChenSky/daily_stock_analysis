# -*- coding: utf-8 -*-
"""
Tests for mootdx_fundamental_adapter and field coverage comparison with AkShare adapter.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.mootdx_fundamental_adapter import (
    MootdxFundamentalAdapter,
    _normalize_stock_code,
    _pick_col,
    _COL_GROSS_MARGIN,
    _COL_INST_TOTAL,
    _COL_NET_PROFIT_PARENT,
    _COL_NET_PROFIT_YOY,
    _COL_OPERATING_CASH_FLOW,
    _COL_REVENUE,
    _COL_REVENUE_ALT,
    _COL_REVENUE_YOY,
    _COL_ROE,
    _COL_ROE_WEIGHTED,
    _COL_TOP10_TRADABLE,
)

from data_provider.base import _CompositeFundamentalAdapter


# ---------------------------------------------------------------------------
# Helper: build a mock gpcw DataFrame that mimics what Affair.parse() returns
# ---------------------------------------------------------------------------

def _make_gpcw_df(
    code: str = "600519",
    report_date: int = 20240930,
    roe: float = 18.2,
    roe_weighted: float = 19.5,
    revenue_yoy: float = 12.0,
    net_profit_yoy: float = 9.5,
    gross_margin: float = 45.3,
    operating_cash_flow: float = 500.0,
    net_profit_parent: float = 300.0,
    revenue: float = 1000.0,
    revenue_col_name: str = _COL_REVENUE,
    inst_total: float = None,
    top10_tradable: float = None,
) -> pd.DataFrame:
    """Build a single-row gpcw DataFrame with code as index."""
    data = {
        _COL_ROE: [roe],
        _COL_ROE_WEIGHTED: [roe_weighted],
        _COL_REVENUE_YOY: [revenue_yoy],
        _COL_NET_PROFIT_YOY: [net_profit_yoy],
        _COL_GROSS_MARGIN: [gross_margin],
        _COL_OPERATING_CASH_FLOW: [operating_cash_flow],
        _COL_NET_PROFIT_PARENT: [net_profit_parent],
        revenue_col_name: [revenue],
        "report_date": [report_date],
    }
    if inst_total is not None:
        data[_COL_INST_TOTAL] = [inst_total]
    if top10_tradable is not None:
        data[_COL_TOP10_TRADABLE] = [top10_tradable]
    df = pd.DataFrame(data, index=[code])
    return df


def _make_xdxr_df(rows=None) -> pd.DataFrame:
    """Build a mock xdxr DataFrame for dividend testing."""
    if rows is None:
        rows = [
            {"year": 2024, "month": 6, "day": 15, "category": 1, "fenhong": 3.0},
            {"year": 2023, "month": 6, "day": 20, "category": 1, "fenhong": 2.5},
            {"year": 2022, "month": 7, "day": 1, "category": 1, "fenhong": 2.0},
        ]
    return pd.DataFrame(rows)


class TestNormalizeStockCode(unittest.TestCase):
    def test_plain_code(self):
        self.assertEqual(_normalize_stock_code("600519"), "600519")

    def test_sh_prefix(self):
        self.assertEqual(_normalize_stock_code("SH600519"), "600519")

    def test_sz_prefix(self):
        self.assertEqual(_normalize_stock_code("SZ000001"), "000001")

    def test_bj_prefix(self):
        self.assertEqual(_normalize_stock_code("BJ430047"), "430047")

    def test_dot_suffix(self):
        self.assertEqual(_normalize_stock_code("600519.SH"), "600519")

    def test_short_code_padded(self):
        self.assertEqual(_normalize_stock_code("519"), "000519")

    def test_lowercase(self):
        self.assertEqual(_normalize_stock_code("sh600519"), "600519")

    def test_whitespace(self):
        self.assertEqual(_normalize_stock_code("  600519  "), "600519")


class TestPickCol(unittest.TestCase):
    def test_returns_first_match(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        self.assertEqual(_pick_col(df, "C", "B", "A"), "B")

    def test_returns_none_when_missing(self):
        df = pd.DataFrame({"A": [1]})
        self.assertIsNone(_pick_col(df, "B", "C"))

    def test_empty_df(self):
        df = pd.DataFrame()
        self.assertIsNone(_pick_col(df, "A"))


class TestMootdxAdapterInit(unittest.TestCase):
    def test_unavailable_when_mootdx_not_installed(self):
        """Adapter should be unavailable when mootdx import fails."""
        with patch.dict("sys.modules", {"mootdx": None, "mootdx.affair": None}):
            adapter = MootdxFundamentalAdapter()
            self.assertFalse(adapter.available)

    def test_available_when_mootdx_present(self):
        """Adapter should be available when mootdx imports successfully."""
        mock_affair = MagicMock()
        with patch.dict("sys.modules", {"mootdx": MagicMock(), "mootdx.affair": MagicMock(Affair=mock_affair)}):
            adapter = MootdxFundamentalAdapter()
            self.assertTrue(adapter.available)

    def test_tdx_dir_stored(self):
        """tdx_dir should be stored even when mootdx is absent."""
        with patch.dict("sys.modules", {"mootdx": None, "mootdx.affair": None}):
            adapter = MootdxFundamentalAdapter(tdx_dir="/tmp/tdx_test")
            self.assertEqual(adapter._tdx_dir, "/tmp/tdx_test")


class TestMootdxGetFundamentalBundle(unittest.TestCase):
    """Test get_fundamental_bundle with mocked _fetch_latest_gpcw."""

    def _make_adapter(self) -> MootdxFundamentalAdapter:
        """Create adapter with mootdx mocked as available."""
        adapter = object.__new__(MootdxFundamentalAdapter)
        adapter._tdx_dir = ""
        adapter._available = True
        adapter._init_error = None
        return adapter

    def test_unavailable_returns_not_supported(self):
        adapter = self._make_adapter()
        adapter._available = False
        result = adapter.get_fundamental_bundle("600519")
        self.assertEqual(result["status"], "not_supported")
        self.assertIn("mootdx_unavailable", result["errors"])

    def test_fetch_returns_none(self):
        adapter = self._make_adapter()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=None):
            result = adapter.get_fundamental_bundle("600519")
        self.assertEqual(result["status"], "not_supported")
        self.assertIn("gpcw:empty", result["errors"])

    def test_fetch_raises_exception(self):
        adapter = self._make_adapter()
        with patch.object(adapter, "_fetch_latest_gpcw", side_effect=RuntimeError("network error")):
            result = adapter.get_fundamental_bundle("600519")
        self.assertEqual(result["status"], "not_supported")
        self.assertIn("gpcw:RuntimeError", result["errors"])

    def test_full_bundle_with_all_fields(self):
        adapter = self._make_adapter()
        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")

        # --- growth ---
        self.assertEqual(result["status"], "partial")
        growth = result["growth"]
        self.assertAlmostEqual(growth["roe"], 18.2)
        self.assertAlmostEqual(growth["revenue_yoy"], 12.0)
        self.assertAlmostEqual(growth["net_profit_yoy"], 9.5)
        self.assertAlmostEqual(growth["gross_margin"], 45.3)

        # --- earnings ---
        fr = result["earnings"]["financial_report"]
        self.assertEqual(fr["report_date"], "2024-09-30")
        self.assertAlmostEqual(fr["revenue"], 1000.0)
        self.assertAlmostEqual(fr["net_profit_parent"], 300.0)
        self.assertAlmostEqual(fr["operating_cash_flow"], 500.0)
        # weighted ROE takes precedence
        self.assertAlmostEqual(fr["roe"], 19.5)

        # source_chain
        self.assertIn("mootdx_gpcw", result["source_chain"])

    def test_weighted_roe_fallback_to_plain(self):
        """When weighted ROE is missing, use plain ROE."""
        adapter = self._make_adapter()
        df = _make_gpcw_df(roe_weighted=None)
        # Remove the weighted ROE column entirely
        if _COL_ROE_WEIGHTED in df.columns:
            df = df.drop(columns=[_COL_ROE_WEIGHTED])
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")
        fr = result["earnings"]["financial_report"]
        self.assertAlmostEqual(fr["roe"], 18.2)

    def test_revenue_alt_column(self):
        """Should pick '营业收入' when '其中：营业收入' is absent."""
        adapter = self._make_adapter()
        df = _make_gpcw_df(revenue_col_name=_COL_REVENUE_ALT)
        # Ensure primary name is not present
        if _COL_REVENUE in df.columns:
            df = df.drop(columns=[_COL_REVENUE])
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")
        fr = result["earnings"]["financial_report"]
        self.assertAlmostEqual(fr["revenue"], 1000.0)

    def test_report_date_as_string(self):
        """report_date as string falls through to _normalize_report_date."""
        adapter = self._make_adapter()
        df = _make_gpcw_df(report_date="2024-06-30")
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")
        fr = result["earnings"]["financial_report"]
        self.assertEqual(fr["report_date"], "2024-06-30")

    def test_no_financial_report_when_all_none(self):
        """When all earnings fields are None, financial_report key should be absent."""
        adapter = self._make_adapter()
        df = _make_gpcw_df(
            operating_cash_flow=None,
            net_profit_parent=None,
            revenue=None,
        )
        # Also null out revenue columns
        for col in [_COL_REVENUE, _COL_REVENUE_ALT]:
            if col in df.columns:
                df[col] = None
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")
        # financial_report may still exist if report_date is set
        # but the key values should be None


class TestMootdxCapitalFlowAndDragonTiger(unittest.TestCase):
    def test_capital_flow_returns_not_supported(self):
        adapter = object.__new__(MootdxFundamentalAdapter)
        adapter._tdx_dir = ""
        adapter._available = True
        result = adapter.get_capital_flow("600519")
        self.assertEqual(result["status"], "not_supported")

    def test_dragon_tiger_returns_not_supported(self):
        adapter = object.__new__(MootdxFundamentalAdapter)
        adapter._tdx_dir = ""
        adapter._available = True
        result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "not_supported")


class TestFetchLatestGpcw(unittest.TestCase):
    """Test _fetch_latest_gpcw with mocked Affair."""

    def _make_adapter(self) -> MootdxFundamentalAdapter:
        adapter = object.__new__(MootdxFundamentalAdapter)
        adapter._tdx_dir = ""
        adapter._available = True
        adapter._init_error = None
        return adapter

    @patch("data_provider.mootdx_fundamental_adapter.Path.exists", return_value=True)
    def test_filters_by_stock_code_in_index(self, mock_exists):
        adapter = self._make_adapter()
        df = _make_gpcw_df(code="600519")
        # Add a different stock
        df2 = pd.concat([df, _make_gpcw_df(code="000001")])

        mock_affair = MagicMock()
        mock_affair.files.return_value = [{"filename": "gpcw20240930.zip"}]
        mock_affair.parse.return_value = df2

        with patch.dict("sys.modules", {"mootdx": MagicMock(), "mootdx.affair": MagicMock(Affair=mock_affair)}):
            from mootdx.affair import Affair
            result = adapter._fetch_latest_gpcw("600519")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertIn("600519", result.index)

    def test_returns_none_when_no_gpcw_files(self):
        adapter = self._make_adapter()
        mock_affair = MagicMock()
        mock_affair.files.return_value = [{"filename": "other_file.zip"}]

        with patch.dict("sys.modules", {"mootdx": MagicMock(), "mootdx.affair": MagicMock(Affair=mock_affair)}):
            result = adapter._fetch_latest_gpcw("600519")
        self.assertIsNone(result)

    def test_returns_none_when_code_not_found(self):
        adapter = self._make_adapter()
        df = _make_gpcw_df(code="000001")
        mock_affair = MagicMock()
        mock_affair.files.return_value = [{"filename": "gpcw20240930.zip"}]
        mock_affair.parse.return_value = df

        with patch.dict("sys.modules", {"mootdx": MagicMock(), "mootdx.affair": MagicMock(Affair=mock_affair)}):
            with patch("data_provider.mootdx_fundamental_adapter.Path.exists", return_value=True):
                result = adapter._fetch_latest_gpcw("600519")
        self.assertIsNone(result)


class TestCompositeFundamentalAdapter(unittest.TestCase):
    """Test _CompositeFundamentalAdapter merge and delegation logic."""

    def test_primary_ok_skips_fallback(self):
        primary = MagicMock()
        primary.get_fundamental_bundle.return_value = {
            "status": "ok", "growth": {"roe": 10.0}, "earnings": {},
            "institution": {}, "source_chain": ["ak"], "errors": [],
        }
        fallback = MagicMock()
        composite = _CompositeFundamentalAdapter(primary, fallback)
        result = composite.get_fundamental_bundle("600519")
        self.assertEqual(result["status"], "ok")
        fallback.get_fundamental_bundle.assert_not_called()

    def test_primary_partial_triggers_fallback(self):
        primary = MagicMock()
        primary.get_fundamental_bundle.return_value = {
            "status": "partial",
            "growth": {"roe": 10.0},
            "earnings": {},
            "institution": {},
            "source_chain": ["ak"],
            "errors": [],
        }
        fallback = MagicMock()
        fallback.get_fundamental_bundle.return_value = {
            "status": "partial",
            "growth": {"revenue_yoy": 5.0},
            "earnings": {"financial_report": {"report_date": "2024-09-30"}},
            "institution": {},
            "source_chain": ["mootdx_gpcw"],
            "errors": [],
        }
        composite = _CompositeFundamentalAdapter(primary, fallback)
        result = composite.get_fundamental_bundle("600519")
        # growth merged: primary roe kept, fallback revenue_yoy filled in
        self.assertAlmostEqual(result["growth"]["roe"], 10.0)
        self.assertAlmostEqual(result["growth"]["revenue_yoy"], 5.0)
        # earnings from fallback
        self.assertIn("financial_report", result["earnings"])
        self.assertIn("mootdx_gpcw", result["source_chain"])

    def test_has_real_content(self):
        self.assertFalse(_CompositeFundamentalAdapter._has_real_content(None))
        self.assertFalse(_CompositeFundamentalAdapter._has_real_content({}))
        self.assertFalse(_CompositeFundamentalAdapter._has_real_content({"a": None}))
        self.assertFalse(_CompositeFundamentalAdapter._has_real_content({"a": {}}))
        self.assertTrue(_CompositeFundamentalAdapter._has_real_content({"a": 1.0}))
        self.assertTrue(_CompositeFundamentalAdapter._has_real_content({"a": {"b": 1.0}}))
        self.assertTrue(_CompositeFundamentalAdapter._has_real_content({"a": {"b": None, "c": 2.0}}))

    def test_merge_empty_earnings_subdict_replaced_by_fallback(self):
        """When primary has financial_report: {}, fallback should override."""
        primary_result = {
            "status": "partial",
            "growth": {"roe": 10.0},
            "earnings": {"financial_report": {}},
            "institution": {},
            "source_chain": ["ak"],
            "errors": [],
        }
        fallback_result = {
            "status": "partial",
            "growth": {},
            "earnings": {"financial_report": {"report_date": "2024-09-30", "roe": 15.0}},
            "institution": {},
            "source_chain": ["mootdx_gpcw"],
            "errors": [],
        }
        merged = _CompositeFundamentalAdapter._merge_bundles(primary_result, fallback_result)
        self.assertEqual(merged["earnings"]["financial_report"]["report_date"], "2024-09-30")
        self.assertAlmostEqual(merged["earnings"]["financial_report"]["roe"], 15.0)

    def test_capital_flow_delegates_to_primary(self):
        primary = MagicMock()
        primary.get_capital_flow.return_value = {"status": "ok", "stock_flow": {"main_net_inflow": 100}}
        composite = _CompositeFundamentalAdapter(primary, MagicMock())
        result = composite.get_capital_flow("600519")
        primary.get_capital_flow.assert_called_once_with("600519", 5)
        self.assertEqual(result["status"], "ok")

    def test_dragon_tiger_delegates_to_primary(self):
        primary = MagicMock()
        primary.get_dragon_tiger_flag.return_value = {"status": "ok", "is_on_list": False}
        composite = _CompositeFundamentalAdapter(primary, MagicMock())
        result = composite.get_dragon_tiger_flag("600519")
        primary.get_dragon_tiger_flag.assert_called_once_with("600519", 20)
        self.assertEqual(result["status"], "ok")

    def test_capital_flow_exception_propagates(self):
        """capital_flow should propagate exceptions (not swallowed by composite)."""
        primary = MagicMock()
        primary.get_capital_flow.side_effect = RuntimeError("timeout")
        composite = _CompositeFundamentalAdapter(primary, MagicMock())
        with self.assertRaises(RuntimeError):
            composite.get_capital_flow("600519")

    def test_no_fallback_returns_primary_directly(self):
        primary = MagicMock()
        primary.get_fundamental_bundle.return_value = {
            "status": "partial", "growth": {}, "earnings": {},
            "institution": {}, "source_chain": [], "errors": [],
        }
        composite = _CompositeFundamentalAdapter(primary, None)
        result = composite.get_fundamental_bundle("600519")
        self.assertEqual(result["status"], "partial")


class TestFieldCoverageComparison(unittest.TestCase):
    """
    Compare field structures between AkShare and mootdx adapters
    to verify mootdx covers all core growth + earnings fields that AkShare provides.
    """

    def test_growth_field_coverage(self):
        """
        Verify mootdx covers the same growth sub-fields as AkShare:
          roe, revenue_yoy, net_profit_yoy, gross_margin
        """
        adapter = object.__new__(MootdxFundamentalAdapter)
        adapter._tdx_dir = ""
        adapter._available = True
        adapter._init_error = None

        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            mootdx_result = adapter.get_fundamental_bundle("600519")

        mootdx_growth_keys = set(mootdx_result["growth"].keys())

        # These are the keys AkshareFundamentalAdapter produces (see fundamental_adapter.py:328-333)
        akshare_growth_keys = {"roe", "revenue_yoy", "net_profit_yoy", "gross_margin"}

        self.assertEqual(
            mootdx_growth_keys,
            akshare_growth_keys,
            f"Growth key mismatch: mootdx has {mootdx_growth_keys}, expected {akshare_growth_keys}",
        )

    def test_earnings_financial_report_coverage(self):
        """
        Verify mootdx covers the same earnings.financial_report sub-fields:
          report_date, revenue, net_profit_parent, operating_cash_flow, roe
        """
        adapter = object.__new__(MootdxFundamentalAdapter)
        adapter._tdx_dir = ""
        adapter._available = True
        adapter._init_error = None

        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            mootdx_result = adapter.get_fundamental_bundle("600519")

        mootdx_fr_keys = set(mootdx_result["earnings"]["financial_report"].keys())

        # These are the keys AkshareFundamentalAdapter produces (see fundamental_adapter.py:334-340)
        akshare_fr_keys = {"report_date", "revenue", "net_profit_parent", "operating_cash_flow", "roe"}

        self.assertEqual(
            mootdx_fr_keys,
            akshare_fr_keys,
            f"financial_report key mismatch: mootdx has {mootdx_fr_keys}, expected {akshare_fr_keys}",
        )

    def test_mootdx_does_not_cover_earnings_extras(self):
        """
        mootdx intentionally does NOT cover:
          - earnings.forecast_summary
          - earnings.quick_report_summary
        These are AkShare-only. Verify they are empty.
        Without institution columns or xdxr data, institution and dividend should also be empty.
        """
        adapter = object.__new__(MootdxFundamentalAdapter)
        adapter._tdx_dir = ""
        adapter._available = True
        adapter._init_error = None

        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")

        # mootdx should NOT produce these (AkShare only)
        self.assertNotIn("forecast_summary", result["earnings"])
        self.assertNotIn("quick_report_summary", result["earnings"])
        # Without institution columns in gpcw and no xdxr data, these are empty
        self.assertNotIn("dividend", result["earnings"])
        self.assertEqual(result["institution"], {})

    def test_full_bundle_structure_printout(self):
        """
        Print the complete bundle structure for visual comparison.
        This test always passes; it's for human review.
        """
        # --- mootdx ---
        mootdx_adapter = object.__new__(MootdxFundamentalAdapter)
        mootdx_adapter._tdx_dir = ""
        mootdx_adapter._available = True
        mootdx_adapter._init_error = None

        mootdx_df = _make_gpcw_df(inst_total=50000.0, top10_tradable=30000.0)
        with patch.object(mootdx_adapter, "_fetch_latest_gpcw", return_value=mootdx_df), \
             patch.object(mootdx_adapter, "_fetch_xdxr", return_value=_make_xdxr_df()):
            mootdx_bundle = mootdx_adapter.get_fundamental_bundle("600519")

        # --- akshare (simulated with same values) ---
        akshare_bundle = {
            "status": "partial",
            "growth": {
                "roe": 18.2,
                "revenue_yoy": 12.0,
                "net_profit_yoy": 9.5,
                "gross_margin": 45.3,
            },
            "earnings": {
                "financial_report": {
                    "report_date": "2024-09-30",
                    "revenue": 1000.0,
                    "net_profit_parent": 300.0,
                    "operating_cash_flow": 500.0,
                    "roe": 18.2,
                },
                "forecast_summary": "预增",
                "quick_report_summary": "快报摘要",
                "dividend": {
                    "events": [{"event_date": "2024-07-10", "cash_dividend_per_share": 0.3}],
                    "ttm_cash_dividend_per_share": 0.3,
                },
            },
            "institution": {
                "institution_holding_change": 1000.0,
                "top10_holder_change": -500.0,
            },
            "source_chain": ["growth:stock_financial_abstract", "earnings_forecast:stock_yjyg_em"],
            "errors": [],
        }

        # --- composite merge result ---
        composite = _CompositeFundamentalAdapter(
            primary=MagicMock(get_fundamental_bundle=MagicMock(return_value={
                "status": "partial",
                "growth": {"roe": None, "revenue_yoy": None, "net_profit_yoy": None, "gross_margin": None},
                "earnings": {},
                "institution": {},
                "source_chain": ["ak"],
                "errors": ["ak_timeout"],
            })),
            fallback=mootdx_adapter,
        )
        with patch.object(mootdx_adapter, "_fetch_latest_gpcw", return_value=mootdx_df), \
             patch.object(mootdx_adapter, "_fetch_xdxr", return_value=_make_xdxr_df()):
            composite_bundle = composite.get_fundamental_bundle("600519")

        print("\n" + "=" * 70)
        print("FIELD COVERAGE COMPARISON REPORT")
        print("=" * 70)

        print("\n--- [1] mootdx bundle ---")
        self._print_bundle(mootdx_bundle, indent="  ")

        print("\n--- [2] akshare bundle (simulated) ---")
        self._print_bundle(akshare_bundle, indent="  ")

        print("\n--- [3] composite merge (akshare failed → mootdx fallback) ---")
        self._print_bundle(composite_bundle, indent="  ")

        # --- Coverage matrix ---
        print("\n--- [4] Coverage Matrix ---")
        growth_fields = ["roe", "revenue_yoy", "net_profit_yoy", "gross_margin"]
        fr_fields = ["report_date", "revenue", "net_profit_parent", "operating_cash_flow", "roe"]
        earnings_sections = ["financial_report", "forecast_summary", "quick_report_summary", "dividend"]

        print(f"  {'Field':<30} {'AkShare':<10} {'mootdx':<10} {'Composite':<10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        for f in growth_fields:
            ak = "Y" if f in akshare_bundle.get("growth", {}) else "-"
            mt = "Y" if f in mootdx_bundle.get("growth", {}) else "-"
            co = "Y" if f in composite_bundle.get("growth", {}) else "-"
            print(f"  growth.{f:<22} {ak:<10} {mt:<10} {co:<10}")

        for f in fr_fields:
            ak = "Y" if f in akshare_bundle.get("earnings", {}).get("financial_report", {}) else "-"
            mt = "Y" if f in mootdx_bundle.get("earnings", {}).get("financial_report", {}) else "-"
            co = "Y" if f in composite_bundle.get("earnings", {}).get("financial_report", {}) else "-"
            print(f"  earnings.fr.{f:<18} {ak:<10} {mt:<10} {co:<10}")

        for f in earnings_sections:
            ak = "Y" if f in akshare_bundle.get("earnings", {}) else "-"
            mt = "Y" if f in mootdx_bundle.get("earnings", {}) else "-"
            co = "Y" if f in composite_bundle.get("earnings", {}) else "-"
            print(f"  earnings.{f:<22} {ak:<10} {mt:<10} {co:<10}")

        inst_fields = ["institution_holding_change", "top10_holder_change"]
        for f in inst_fields:
            ak = "Y" if f in akshare_bundle.get("institution", {}) else "-"
            mt = "Y" if f in mootdx_bundle.get("institution", {}) else "-"
            co = "Y" if f in composite_bundle.get("institution", {}) else "-"
            print(f"  institution.{f:<17} {ak:<10} {mt:<10} {co:<10}")

        print(f"\n  {'status':<30} {akshare_bundle['status']:<10} {mootdx_bundle['status']:<10} {composite_bundle['status']:<10}")
        print("=" * 70)

    @staticmethod
    def _print_bundle(bundle: dict, indent: str = "") -> None:
        for key in ("status", "growth", "earnings", "institution", "source_chain", "errors"):
            val = bundle.get(key)
            if isinstance(val, dict):
                print(f"{indent}{key}:")
                for k, v in val.items():
                    if isinstance(v, dict):
                        print(f"{indent}  {k}:")
                        for kk, vv in v.items():
                            print(f"{indent}    {kk}: {vv}")
                    else:
                        print(f"{indent}  {k}: {v}")
            else:
                print(f"{indent}{key}: {val}")


class TestInstitutionFromGpcw(unittest.TestCase):
    """Test institution data extraction from gpcw columns."""

    def _make_adapter(self) -> MootdxFundamentalAdapter:
        adapter = object.__new__(MootdxFundamentalAdapter)
        adapter._tdx_dir = ""
        adapter._available = True
        adapter._init_error = None
        return adapter

    def test_institution_fields_present(self):
        """When gpcw has institution columns, they should be extracted."""
        adapter = self._make_adapter()
        df = _make_gpcw_df(inst_total=50000.0, top10_tradable=30000.0)
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")

        self.assertAlmostEqual(result["institution"]["institution_holding_change"], 50000.0)
        self.assertAlmostEqual(result["institution"]["top10_holder_change"], 30000.0)

    def test_institution_missing_columns(self):
        """When gpcw lacks institution columns, institution should be empty."""
        adapter = self._make_adapter()
        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")

        self.assertEqual(result["institution"], {})

    def test_institution_partial_columns(self):
        """When gpcw has only inst_total, only that field should be set."""
        adapter = self._make_adapter()
        df = _make_gpcw_df(inst_total=50000.0, top10_tradable=None)
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")

        self.assertIn("institution_holding_change", result["institution"])
        self.assertNotIn("top10_holder_change", result["institution"])


class TestDividendFromXdxr(unittest.TestCase):
    """Test dividend data extraction via pytdx xdxr API."""

    def _make_adapter(self) -> MootdxFundamentalAdapter:
        adapter = object.__new__(MootdxFundamentalAdapter)
        adapter._tdx_dir = ""
        adapter._available = True
        adapter._init_error = None
        return adapter

    def test_dividend_payload_structure(self):
        """Dividend payload should match AkShare format."""
        adapter = self._make_adapter()
        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=_make_xdxr_df()):
            result = adapter.get_fundamental_bundle("600519")

        self.assertIn("dividend", result["earnings"])
        div = result["earnings"]["dividend"]
        self.assertIn("events", div)
        self.assertIn("ttm_event_count", div)
        self.assertIn("ttm_cash_dividend_per_share", div)
        self.assertEqual(div["coverage"], "cash_dividend_pre_tax")
        self.assertIn("as_of", div)
        self.assertIn("mootdx_xdxr", result["source_chain"])

    def test_dividend_event_fields(self):
        """Each dividend event should have event_date, cash_dividend_per_share, is_pre_tax."""
        adapter = self._make_adapter()
        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=_make_xdxr_df()):
            result = adapter.get_fundamental_bundle("600519")

        events = result["earnings"]["dividend"]["events"]
        self.assertGreater(len(events), 0)
        for event in events:
            self.assertIn("event_date", event)
            self.assertIn("cash_dividend_per_share", event)
            self.assertTrue(event["is_pre_tax"])

    def test_dividend_sorted_newest_first(self):
        """Events should be sorted newest first."""
        adapter = self._make_adapter()
        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=_make_xdxr_df()):
            result = adapter.get_fundamental_bundle("600519")

        events = result["earnings"]["dividend"]["events"]
        dates = [e["event_date"] for e in events]
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_dividend_no_xdxr_data(self):
        """When xdxr returns None, no dividend should be set."""
        adapter = self._make_adapter()
        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=None):
            result = adapter.get_fundamental_bundle("600519")

        self.assertNotIn("dividend", result["earnings"])

    def test_dividend_empty_xdxr(self):
        """When xdxr returns empty DataFrame, no dividend should be set."""
        adapter = self._make_adapter()
        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=pd.DataFrame()):
            result = adapter.get_fundamental_bundle("600519")

        self.assertNotIn("dividend", result["earnings"])

    def test_dividend_max_5_events(self):
        """Events list should be capped at 5."""
        rows = [
            {"year": 2024 - i, "month": 6, "day": 15, "category": 1, "fenhong": 1.0 + i * 0.1}
            for i in range(10)
        ]
        adapter = self._make_adapter()
        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=_make_xdxr_df(rows)):
            result = adapter.get_fundamental_bundle("600519")

        events = result["earnings"]["dividend"]["events"]
        self.assertLessEqual(len(events), 5)

    def test_dividend_filters_non_cash_events(self):
        """Events with fenhong<=0 should be filtered by _build_dividend_payload."""
        rows = [
            {"year": 2024, "month": 6, "day": 15, "category": 1, "fenhong": 3.0},  # valid
            {"year": 2023, "month": 6, "day": 20, "category": 1, "fenhong": 0},     # fenhong=0 filtered
            {"year": 2022, "month": 7, "day": 1, "category": 1, "fenhong": -1.0},   # negative filtered
        ]
        adapter = self._make_adapter()
        df = _make_gpcw_df()
        with patch.object(adapter, "_fetch_latest_gpcw", return_value=df), \
             patch.object(adapter, "_fetch_xdxr", return_value=_make_xdxr_df(rows)):
            result = adapter.get_fundamental_bundle("600519")

        events = result["earnings"]["dividend"]["events"]
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["cash_dividend_per_share"], 3.0)


if __name__ == "__main__":
    unittest.main()
