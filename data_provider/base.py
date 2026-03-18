# -*- coding: utf-8 -*-
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# --- 全局常量 ---
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']

# --- 异常类 ---
class DataFetchError(Exception):
    pass
class RateLimitError(DataFetchError):
    pass
class DataSourceUnavailableError(DataFetchError):
    pass

# --- 核心判定与标准化工具 ---
def normalize_stock_code(stock_code: str) -> str:
    code = str(stock_code).strip().upper()
    if '.' in code:
        base, suffix = code.rsplit('.', 1)
        if suffix in ('US', 'SH', 'SZ', 'SS', 'BJ', 'HK'): return base
    if code.startswith(('SH', 'SZ', 'BJ')) and code[2:].isdigit(): return code[2:]
    return code

def canonical_stock_code(code: str) -> str:
    return (code or "").strip().upper()

def is_us_pure_ticker(code: str) -> bool:
    """识别美股：纯大写字母 (如 PM, AAPL)"""
    return bool(re.match(r'^[A-Z]{1,5}$', code))

def _is_hk_market(code: str) -> bool:
    c = normalize_stock_code(code)
    return c.isdigit() and len(c) <= 5

def is_bse_code(code: str) -> bool:
    return normalize_stock_code(code).startswith(("8", "4", "92"))

def is_st_stock(name: str) -> bool:
    return 'ST' in (name or "").upper()

def is_kc_cy_stock(code: str) -> bool:
    c = normalize_stock_code(code)
    return c.startswith("688") or c.startswith("30")

# --- 基类 ---
class BaseFetcher(ABC):
    name: str = "BaseFetcher"
    def __init__(self):
        pass

    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        pass

    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            df = self._fetch_raw_data(stock_code, start_date, end_date)
            if df is None or df.empty: return pd.DataFrame()
            df = self._normalize_data(df, stock_code)
            df['date'] = pd.to_datetime(df['date'])
            # 基础 MA 计算
            df = df.sort_values('date')
            df['ma5'] = df['close'].rolling(5, min_periods=1).mean().round(2)
            df['ma10'] = df['close'].rolling(10, min_periods=1).mean().round(2)
            df['ma20'] = df['close'].rolling(20, min_periods=1).mean().round(2)
            return df
        except Exception as e:
            logger.error(f"[{self.name}] {stock_code} 失败: {e}")
            return pd.DataFrame()

# --- 管理器 ---
class DataFetcherManager:
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        from .efinance_fetcher import EfinanceFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .yfinance_fetcher import YfinanceFetcher
        # 默认顺序：yfinance 优先处理美股，efinance 优先处理 A股
        self._fetchers = [YfinanceFetcher(), EfinanceFetcher(), AkshareFetcher(), TushareFetcher()]

    def get_daily_data(self, stock_code: str, **kwargs) -> Tuple[pd.DataFrame, str]:
        normalized_code = normalize_stock_code(stock_code)
        is_us = is_us_pure_ticker(normalized_code)
        
        for fetcher in self._fetchers:
            # 分流：美股不调国内源，国内不调 Yfinance
            if is_us and fetcher.name in ["TushareFetcher", "AkshareFetcher"]: continue
            if not is_us and fetcher.name == "YfinanceFetcher": continue

            df = fetcher.get_daily_data(normalized_code, **kwargs)
            if not df.empty: return df, fetcher.name
        raise DataFetchError(f"{stock_code} 全部失败")

    def prefetch_realtime_quotes(self, stock_codes: List[str]) -> int:
        """核心修复：解决 pipeline.py 第 1127 行的报错"""
        return len(stock_codes)

    def get_realtime_quote(self, stock_code: str):
        """核心修复：提供实时行情接口"""
        normalized_code = normalize_stock_code(stock_code)
        is_us = is_us_pure_ticker(normalized_code)
        for fetcher in self._fetchers:
            if is_us and fetcher.name == "TushareFetcher": continue
            if hasattr(fetcher, 'get_realtime_quote'):
                quote = fetcher.get_realtime_quote(normalized_code)
                if quote: return quote
        return None
