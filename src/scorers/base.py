# -*- coding: utf-8 -*-
"""Base interface for dimension scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DimensionScore:
    """Structured score result for a single evaluation dimension."""

    dimension: str                    # "technical" | "chip" | "capital_flow" | "fundamental"
    score: Optional[float] = None    # 0-100, None when data is missing
    weight: float = 1.0              # dynamic weight (set to 0 when data missing)
    data_status: str = "ok"          # "ok" | "partial" | "missing"
    sub_scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 1) if self.score is not None else None,
            "weight": self.weight,
            "data_status": self.data_status,
            "sub_scores": {k: round(v, 1) for k, v in self.sub_scores.items()},
            "reasons": self.reasons,
            "risk_flags": self.risk_flags,
        }

    @property
    def label(self) -> str:
        """Human-readable label for display in prompt."""
        _LABELS = {
            "technical": "技术面",
            "chip": "筹码结构",
            "capital_flow": "资金面",
            "fundamental": "基本面",
        }
        return _LABELS.get(self.dimension, self.dimension)

    @property
    def status_label(self) -> str:
        """Human-readable status assessment."""
        if self.score is None:
            return "数据缺失"
        if self.score >= 70:
            return "偏多"
        if self.score >= 40:
            return "中性"
        return "偏空"


class BaseDimensionScorer(ABC):
    """Abstract base class for all dimension scorers."""

    dimension_name: str = ""

    @abstractmethod
    def score(self, context: Dict[str, object]) -> DimensionScore:
        """Evaluate the dimension based on pipeline context.

        Args:
            context: The enhanced_context dict from pipeline._enhance_context(),
                     containing keys like 'chip', 'trend_analysis',
                     'fundamental_context', 'realtime', etc.
        """
        ...
