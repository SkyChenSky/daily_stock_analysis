# -*- coding: utf-8 -*-
"""Capital flow dimension scorer.

Evaluates money flow based on:
- Main net inflow direction (主力净流入方向)
- 5-day capital trend (5日资金趋势)
- 10-day capital trend (10日资金趋势)
- Sector resonance (板块资金共振)
"""

from __future__ import annotations

from typing import Dict, Optional

from src.scorers.base import BaseDimensionScorer, DimensionScore


class CapitalFlowScorer(BaseDimensionScorer):
    """Score capital flow from fundamental_context.capital_flow data."""

    dimension_name = "capital_flow"

    def score(self, context: Dict[str, object]) -> DimensionScore:
        fundamental_ctx = context.get("fundamental_context")
        if not isinstance(fundamental_ctx, dict):
            return DimensionScore(
                dimension=self.dimension_name,
                score=None,
                weight=0,
                data_status="missing",
                reasons=["资金流数据缺失"],
            )

        cf_block = fundamental_ctx.get("capital_flow", {})
        if not isinstance(cf_block, dict):
            return DimensionScore(
                dimension=self.dimension_name,
                score=None,
                weight=0,
                data_status="missing",
            )

        cf_status = cf_block.get("status", "not_supported")
        if cf_status in ("not_supported", "failed"):
            return DimensionScore(
                dimension=self.dimension_name,
                score=None,
                weight=0,
                data_status="missing",
                reasons=["资金流数据不可用"],
            )

        cf_data = cf_block.get("data", {})
        if not isinstance(cf_data, dict) or not cf_data:
            return DimensionScore(
                dimension=self.dimension_name,
                score=None,
                weight=0,
                data_status="missing",
            )

        stock_flow = cf_data.get("stock_flow", {})
        sector_rank = cf_data.get("sector_rankings", {})
        sub_scores: Dict[str, float] = {}
        reasons: list[str] = []
        risk_flags: list[str] = []

        # 1. Main net inflow direction (25 points)
        if isinstance(stock_flow, dict):
            net_inflow = stock_flow.get("main_net_inflow")
            if isinstance(net_inflow, (int, float)):
                net_inflow = float(net_inflow)
                if net_inflow > 0:
                    sub_scores["main_inflow"] = 25
                    reasons.append(f"主力净流入({net_inflow:.0f}万)，资金做多")
                elif net_inflow > -500:
                    sub_scores["main_inflow"] = 12
                else:
                    sub_scores["main_inflow"] = 0
                    risk_flags.append(f"主力大幅净流出({net_inflow:.0f}万)")
            else:
                sub_scores["main_inflow"] = 12

            # 2. 5-day trend (25 points)
            flow_5d = stock_flow.get("inflow_5d")
            if isinstance(flow_5d, (int, float)):
                flow_5d = float(flow_5d)
                if flow_5d > 0:
                    sub_scores["flow_5d"] = 25
                    reasons.append("近5日资金持续流入")
                elif flow_5d > -300:
                    sub_scores["flow_5d"] = 12
                else:
                    sub_scores["flow_5d"] = 0
                    risk_flags.append("近5日资金持续流出")
            else:
                sub_scores["flow_5d"] = 12

            # 3. 10-day trend (25 points)
            flow_10d = stock_flow.get("inflow_10d")
            if isinstance(flow_10d, (int, float)):
                flow_10d = float(flow_10d)
                if flow_10d > 0:
                    sub_scores["flow_10d"] = 25
                    reasons.append("近10日中期资金方向偏多")
                elif flow_10d > -500:
                    sub_scores["flow_10d"] = 12
                else:
                    sub_scores["flow_10d"] = 0
            else:
                sub_scores["flow_10d"] = 12
        else:
            sub_scores["main_inflow"] = 12
            sub_scores["flow_5d"] = 12
            sub_scores["flow_10d"] = 12

        # 4. Sector resonance (25 points)
        if isinstance(sector_rank, dict):
            top_sectors = sector_rank.get("top", [])
            if isinstance(top_sectors, list) and len(top_sectors) > 0:
                sub_scores["sector_resonance"] = 20
                top_names = [s.get("name", "") for s in top_sectors[:3] if isinstance(s, dict)]
                if top_names:
                    reasons.append(f"所属板块资金流入靠前：{', '.join(top_names)}")
            else:
                sub_scores["sector_resonance"] = 10
        else:
            sub_scores["sector_resonance"] = 12

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
