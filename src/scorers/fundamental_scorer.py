# -*- coding: utf-8 -*-
"""Fundamental dimension scorer.

Evaluates company fundamentals based on:
- Profitability (ROE)
- Growth (revenue YoY + net profit YoY)
- Earnings quality (operating cash flow vs net profit)
- Valuation reasonableness (PE ratio)
"""

from __future__ import annotations

from typing import Dict, Optional

from src.scorers.base import BaseDimensionScorer, DimensionScore


class FundamentalScorer(BaseDimensionScorer):
    """Score fundamentals from fundamental_context data."""

    dimension_name = "fundamental"

    def score(self, context: Dict[str, object]) -> DimensionScore:
        fundamental_ctx = context.get("fundamental_context")
        if not isinstance(fundamental_ctx, dict):
            return DimensionScore(
                dimension=self.dimension_name,
                score=None,
                weight=0,
                data_status="missing",
                reasons=["基本面数据缺失"],
            )

        growth_block = fundamental_ctx.get("growth", {})
        earnings_block = fundamental_ctx.get("earnings", {})

        # Extract growth data
        growth_data = growth_block.get("data", {}) if isinstance(growth_block, dict) else {}
        growth_status = growth_block.get("status", "not_supported") if isinstance(growth_block, dict) else "not_supported"

        # Extract earnings data for cash flow
        earnings_data = earnings_block.get("data", {}) if isinstance(earnings_block, dict) else {}
        financial_report = earnings_data.get("financial_report", {}) if isinstance(earnings_data, dict) else {}

        # Check if any meaningful data exists
        has_growth = isinstance(growth_data, dict) and any(
            growth_data.get(k) is not None for k in ("revenue_yoy", "net_profit_yoy", "roe", "gross_margin")
        )
        has_earnings = isinstance(financial_report, dict) and any(
            financial_report.get(k) is not None for k in ("roe", "operating_cash_flow")
        )

        if not has_growth and not has_earnings:
            return DimensionScore(
                dimension=self.dimension_name,
                score=None,
                weight=0,
                data_status="missing",
                reasons=["基本面数据不可用"],
            )

        sub_scores: Dict[str, float] = {}
        reasons: list[str] = []
        risk_flags: list[str] = []

        # 1. Profitability — ROE (25 points)
        roe = None
        if isinstance(growth_data, dict):
            roe = growth_data.get("roe")
        if roe is None and isinstance(financial_report, dict):
            roe = financial_report.get("roe")

        if isinstance(roe, (int, float)):
            roe = float(roe)
            if roe > 20:
                sub_scores["roe"] = 25
                reasons.append(f"ROE优秀({roe:.1f}%)")
            elif roe > 15:
                sub_scores["roe"] = 20
                reasons.append(f"ROE良好({roe:.1f}%)")
            elif roe > 10:
                sub_scores["roe"] = 12
            elif roe > 0:
                sub_scores["roe"] = 5
            else:
                sub_scores["roe"] = 0
                risk_flags.append(f"ROE为负({roe:.1f}%)，盈利能力堪忧")
        else:
            sub_scores["roe"] = 12

        # 2. Growth (25 points)
        rev_yoy = growth_data.get("revenue_yoy") if isinstance(growth_data, dict) else None
        profit_yoy = growth_data.get("net_profit_yoy") if isinstance(growth_data, dict) else None

        rev_positive = isinstance(rev_yoy, (int, float)) and float(rev_yoy) > 0
        profit_positive = isinstance(profit_yoy, (int, float)) and float(profit_yoy) > 0

        if rev_positive and profit_positive:
            sub_scores["growth"] = 25
            rev_str = f"{float(rev_yoy):.1f}%" if rev_yoy is not None else "N/A"
            profit_str = f"{float(profit_yoy):.1f}%" if profit_yoy is not None else "N/A"
            reasons.append(f"营收同比{rev_str}，净利同比{profit_str}，双增长")
        elif rev_positive or profit_positive:
            sub_scores["growth"] = 15
        elif isinstance(rev_yoy, (int, float)) and isinstance(profit_yoy, (int, float)):
            sub_scores["growth"] = 0
            risk_flags.append("营收与净利润同比双降")
        else:
            sub_scores["growth"] = 12

        # 3. Earnings quality — cash flow vs net profit (25 points)
        cash_flow = financial_report.get("operating_cash_flow") if isinstance(financial_report, dict) else None
        net_profit = financial_report.get("net_profit_parent") if isinstance(financial_report, dict) else None

        if isinstance(cash_flow, (int, float)) and float(cash_flow) > 0:
            sub_scores["cash_quality"] = 20
            if isinstance(net_profit, (int, float)) and float(net_profit) > 0:
                cf_ratio = float(cash_flow) / float(net_profit)
                if cf_ratio >= 0.8:
                    sub_scores["cash_quality"] = 25
                    reasons.append("经营现金流匹配净利润，盈利质量高")
                elif cf_ratio >= 0.5:
                    sub_scores["cash_quality"] = 15
                else:
                    sub_scores["cash_quality"] = 8
                    risk_flags.append("经营现金流低于净利润，盈利质量存疑")
        elif isinstance(cash_flow, (int, float)) and float(cash_flow) <= 0:
            sub_scores["cash_quality"] = 0
            risk_flags.append("经营现金流为负")
        else:
            sub_scores["cash_quality"] = 12

        # 4. Valuation — PE (25 points, simple rules)
        realtime = context.get("realtime", {})
        pe = realtime.get("pe_ratio") if isinstance(realtime, dict) else None

        if isinstance(pe, (int, float)):
            pe = float(pe)
            if pe <= 0:
                sub_scores["valuation"] = 5
                risk_flags.append("PE为负，公司亏损")
            elif pe < 15:
                sub_scores["valuation"] = 25
                reasons.append(f"PE较低({pe:.1f})，估值有吸引力")
            elif pe < 30:
                sub_scores["valuation"] = 18
            elif pe < 60:
                sub_scores["valuation"] = 10
            else:
                sub_scores["valuation"] = 3
                risk_flags.append(f"PE偏高({pe:.1f})，估值风险")
        else:
            sub_scores["valuation"] = 12

        total = sum(sub_scores.values())
        data_status = "partial" if len(sub_scores) < 4 else "ok"

        return DimensionScore(
            dimension=self.dimension_name,
            score=total,
            weight=1.0,
            data_status=data_status,
            sub_scores=sub_scores,
            reasons=reasons,
            risk_flags=risk_flags,
        )
