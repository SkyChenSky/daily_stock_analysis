# -*- coding: utf-8 -*-
"""Chip structure dimension scorer.

Evaluates chip distribution health based on:
- Profit ratio (获利比例)
- Concentration (筹码集中度)
- Chip health status (筹码状态)
- Price vs average cost (价格与平均成本关系)
"""

from __future__ import annotations

from typing import Dict, Optional

from src.scorers.base import BaseDimensionScorer, DimensionScore


class ChipScorer(BaseDimensionScorer):
    """Score chip structure quality from ChipDistribution data."""

    dimension_name = "chip"

    def score(self, context: Dict[str, object]) -> DimensionScore:
        chip = context.get("chip")
        if not chip or not isinstance(chip, dict):
            return DimensionScore(
                dimension=self.dimension_name,
                score=None,
                weight=0,
                data_status="missing",
                reasons=["筹码数据缺失"],
            )

        sub_scores: Dict[str, float] = {}
        reasons: list[str] = []
        risk_flags: list[str] = []

        # 1. Profit ratio (25 points)
        profit_ratio = chip.get("profit_ratio", 0)
        if isinstance(profit_ratio, (int, float)):
            profit_ratio = float(profit_ratio)
            if profit_ratio < 0.5:
                sub_scores["profit_ratio"] = 22
                reasons.append("获利比例低(<50%)，套牢盘多，抛压轻")
            elif profit_ratio <= 0.7:
                sub_scores["profit_ratio"] = 20
                reasons.append("获利比例适中(50-70%)")
            elif profit_ratio <= 0.9:
                sub_scores["profit_ratio"] = 10
                risk_flags.append(f"获利比例偏高({profit_ratio:.0%})，获利盘抛压风险")
            else:
                sub_scores["profit_ratio"] = 5
                risk_flags.append(f"获利比例极高({profit_ratio:.0%})，抛压风险大")
        else:
            sub_scores["profit_ratio"] = 10

        # 2. Concentration (25 points) — use concentration_90
        concentration = chip.get("concentration_90")
        if isinstance(concentration, (int, float)):
            concentration = float(concentration)
            if concentration < 0.10:
                sub_scores["concentration"] = 25
                reasons.append("筹码高度集中(<10%)，主力控盘")
            elif concentration <= 0.15:
                sub_scores["concentration"] = 20
                reasons.append("筹码较集中(10-15%)")
            elif concentration <= 0.25:
                sub_scores["concentration"] = 12
            else:
                sub_scores["concentration"] = 5
                risk_flags.append(f"筹码分散(>{concentration:.0%})，缺乏主力控盘")
        else:
            sub_scores["concentration"] = 12

        # 3. Chip health status (25 points)
        chip_status = chip.get("chip_status", "")
        status_score_map = {
            "筹码集中": 25,
            "筹码趋集": 20,
            "筹码分散": 8,
            "筹码趋散": 5,
        }
        matched = False
        if isinstance(chip_status, str):
            for key, val in status_score_map.items():
                if key in chip_status:
                    sub_scores["chip_health"] = val
                    if val >= 20:
                        reasons.append(f"筹码状态：{chip_status}")
                    matched = True
                    break
        if not matched:
            sub_scores["chip_health"] = 15

        # 4. Price vs average cost (25 points)
        avg_cost = chip.get("avg_cost")
        realtime = context.get("realtime", {})
        current_price = realtime.get("price") if isinstance(realtime, dict) else None
        if (
            avg_cost is not None
            and current_price is not None
            and isinstance(avg_cost, (int, float))
            and isinstance(current_price, (int, float))
            and float(avg_cost) > 0
        ):
            premium = (float(current_price) - float(avg_cost)) / float(avg_cost)
            if 0.05 <= premium <= 0.15:
                sub_scores["price_vs_cost"] = 25
                reasons.append(f"现价高于平均成本{premium:.1%}，位置合理")
            elif 0 <= premium < 0.05:
                sub_scores["price_vs_cost"] = 20
                reasons.append("现价贴近平均成本，支撑较强")
            elif premium > 0.15:
                sub_scores["price_vs_cost"] = 10
            else:
                sub_scores["price_vs_cost"] = 8
                risk_flags.append(f"现价低于平均成本({premium:.1%})，套牢盘压力大")
        else:
            sub_scores["price_vs_cost"] = 12

        total = sum(sub_scores.values())
        return DimensionScore(
            dimension=self.dimension_name,
            score=total,
            weight=1.0,
            data_status="ok",
            sub_scores=sub_scores,
            reasons=reasons,
            risk_flags=risk_flags,
        )
