# -*- coding: utf-8 -*-
from __future__ import annotations

"""
===================================
股票代码与名称映射
===================================

Shared stock code -> name mapping, used by analyzer, data_provider, and name_to_code_resolver.
"""

# Stock code -> name mapping (common stocks)
STOCK_NAME_MAP = {
    # === A-shares ===
    # A-share names are dynamically resolved via data providers (Baostock/Akshare).
    # Add hardcoded mappings here only if dynamic resolution fails.

    # === US stocks ===
    "PM": "菲利普莫里斯",
    "ENB": "恩桥",
    "ADUS": "阿迪斯",
    "MMM": "3M",
    "MSFT": "微软",
    "COST": "开市客",
    "ABBV": "艾伯维",
    "GLD": "黄金ETF-SPDR",
    "PFE": "辉瑞",
    "O": "Realty Income",
    "BIOA": "BioAge Labs",
    "VZ": "威瑞森",
    "AAPL": "苹果",
    "TSLA": "特斯拉",
    "NVDA": "英伟达",
    "GOOGL": "谷歌A",
    "GOOG": "谷歌C",
    "AMZN": "亚马逊",
    "META": "Meta",
    "AMD": "AMD",
    "INTC": "英特尔",
    "BABA": "阿里巴巴",
    "PDD": "拼多多",
    "JD": "京东",
    "BIDU": "百度",
    "NIO": "蔚来",
    "XPEV": "小鹏汽车",
    "LI": "理想汽车",
    "COIN": "Coinbase",
    "MSTR": "MicroStrategy",

    # === HK stocks (5-digit) ===
    # No HK stocks in current portfolio.
}


def is_meaningful_stock_name(name: str | None, stock_code: str) -> bool:
    """Return whether a stock name is useful for display or caching."""
    if not name:
        return False

    normalized_name = str(name).strip()
    if not normalized_name:
        return False

    normalized_code = (stock_code or "").strip().upper()
    if normalized_name.upper() == normalized_code:
        return False

    if normalized_name.startswith("股票"):
        return False

    placeholder_values = {
        "N/A",
        "NA",
        "NONE",
        "NULL",
        "--",
        "-",
        "UNKNOWN",
        "TICKER",
    }
    if normalized_name.upper() in placeholder_values:
        return False

    return True
