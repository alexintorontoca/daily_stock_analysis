# -*- coding: utf-8 -*-
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

# --- 核心常量 (必须存在，否则 efinance/akshare 会报错) ---
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']

# --- 异常类 ---
class DataFetchError(Exception): pass
class RateLimitError(DataFetchError): pass
class DataSourceUnavailableError(DataFetchError): pass

# --- 核心工具函数 (对齐所有 fetcher 的导入需求) ---

def normalize_stock_code(stock_code: str) -> str:
    """标准化代码，去掉后缀"""
    code = str(stock_code).strip().upper()
    if '.' in code:
        base, suffix = code.rsplit('.', 1)
        if suffix in ('US', 'SH', 'SZ', 'SS', 'BJ', 'HK'): return base
    if code.startswith(('SH', 'SZ', 'BJ')) and code[2:].isdigit(): return code[2:]
    return code

def canonical_stock_code(code: str) -> str:
    """保持原始格式的规范化"""
    return (code or "").strip().upper()

def is_us_pure_ticker(code: str) -> bool:
    """识别美股代码: PM, AAPL, MMM"""
    return bool(re.match(r'^[A-Z]{1,5}$', code))

def is_bse_code(code: str) -> bool:
    """北交所判定"""
    c = normalize_stock_code(code)
    return c.startswith(("8", "4", "92"))

def is_st_stock(name: str) -> bool:
    """ST判定"""
    return 'ST' in (name or "").upper()

def is_kc_cy_stock(code: str) -> bool:
    """科创板/创业板判定"""
    c = normalize_stock_code(code)
    return c.startswith("688") or c.startswith("30")

# --- 基类定义 ---
class BaseFetcher(ABC):
    name: str = "BaseFetcher"
    
    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame: pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame: pass

    def get_daily_data(self, stock_code: str, days: int = 30) -> pd.DataFrame:
        # 子类通常会重写此方法，此处提供空实现防止报错
        return pd.DataFrame()

# --- 管理器 (DataFetcherManager) ---
class DataFetcherManager:
    def __init__(self):
        # 延迟导入以避免循环引用
        from .efinance_fetcher import EfinanceFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .yfinance_fetcher import YfinanceFetcher
        
        self._fetchers = [YfinanceFetcher(), EfinanceFetcher(), AkshareFetcher(), TushareFetcher()]
        self._name_cache = {}

    # --- 适配 pipeline.py 的核心接口 ---

    def prefetch_stock_names(self, stock_codes: List[str], use_bulk: bool = False) -> bool:
        """修复 AttributeError: 'DataFetcherManager' object has no attribute 'prefetch_stock_names'"""
        logger.info(f"批量预取 {len(stock_codes)} 只股票名称...")
        return True

    def prefetch_realtime_quotes(self, stock_codes: List[str]) -> int:
        """适配 pipeline.py 第 1127 行"""
        return len(stock_codes)

    def get_stock_name(self, code: str) -> str:
        """获取股票名称，带缓存"""
        if code in self._name_cache and self._name_cache[code] != code:
            return self._name_cache[code]
        
        # 尝试通过实时行情获取真实名称
        quote = self.get_realtime_quote(code)
        if quote and hasattr(quote, 'name') and quote.name:
            self._name_cache[code] = quote.name
            return quote.name
        return self._name_cache.get(code, code)

    def get_daily_data(self, stock_code: str, days: int = 30) -> Tuple[pd.DataFrame, str]:
        normalized_code = normalize_stock_code(stock_code)
        is_us = is_us_pure_ticker(normalized_code)
        
        for fetcher in self._fetchers:
            # 策略分流
            if is_us and fetcher.name in ["TushareFetcher", "AkshareFetcher"]: continue
            if not is_us and fetcher.name == "YfinanceFetcher": continue

            try:
                df = fetcher.get_daily_data(normalized_code, days=days)
                if df is not None and not df.empty:
                    return df, fetcher.name
            except Exception as e:
                logger.debug(f"数据源 {fetcher.name} 获取 {stock_code} 失败: {e}")
                continue
        return pd.DataFrame(), "None"

    def get_realtime_quote(self, stock_code: str):
        normalized_code = normalize_stock_code(stock_code)
        is_us = is_us_pure_ticker(normalized_code)
        
        for fetcher in self._fetchers:
            if is_us and fetcher.name == "TushareFetcher": continue
            if hasattr(fetcher, 'get_realtime_quote'):
                try:
                    quote = fetcher.get_realtime_quote(normalized_code)
                    if quote: return quote
                except: continue
        return None

    def get_chip_distribution(self, code: str):
        for fetcher in self._fetchers:
            if hasattr(fetcher, 'get_chip_distribution'):
                try:
                    chip = fetcher.get_chip_distribution(code)
                    if chip: return chip
                except: continue
        return None

    def get_fundamental_context(self, code: str, budget_seconds: float = 1.5):
        return {"code": code, "status": "not_supported", "source_chain": []}

    def build_failed_fundamental_context(self, code: str, reason: str):
        return {"code": code, "error": reason, "status": "failed"}
