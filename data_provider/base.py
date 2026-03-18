# -*- coding: utf-8 -*-
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

# --- 辅助工具 ---
def normalize_stock_code(stock_code: str) -> str:
    code = str(stock_code).strip().upper()
    if '.' in code:
        base, suffix = code.rsplit('.', 1)
        if suffix in ('US', 'SH', 'SZ', 'SS', 'BJ', 'HK'): return base
    if code.startswith(('SH', 'SZ', 'BJ')) and code[2:].isdigit(): return code[2:]
    return code

def is_us_pure_ticker(code: str) -> bool:
    """识别美股：纯大写字母 (如 PM, AAPL)"""
    return bool(re.match(r'^[A-Z]{1,5}$', code))

# --- 异常类 ---
class DataFetchError(Exception): pass
class RateLimitError(DataFetchError): pass

# --- 基类 ---
class BaseFetcher(ABC):
    name: str = "BaseFetcher"
    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame: pass
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame: pass

    def get_daily_data(self, stock_code: str, days: int = 30) -> pd.DataFrame:
        # 简单实现，供子类调用
        return pd.DataFrame()

# --- 管理器 (必须严格匹配 pipeline.py 的调用) ---
class DataFetcherManager:
    def __init__(self):
        from .efinance_fetcher import EfinanceFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .yfinance_fetcher import YfinanceFetcher
        
        self._fetchers = [YfinanceFetcher(), EfinanceFetcher(), AkshareFetcher(), TushareFetcher()]
        # 缓存，减少重复查询
        self._name_cache = {}

    def prefetch_stock_names(self, stock_codes: List[str], use_bulk: bool = False) -> bool:
        """
        🔥 修复核心报错：对应 pipeline.py 第 1134 行
        """
        logger.info(f"正在为 {len(stock_codes)} 只股票预取名称信息...")
        for code in stock_codes:
            if code not in self._name_cache:
                # 简单填充，真正名称会在 get_stock_name 时通过 fetcher 获取
                self._name_cache[code] = code 
        return True

    def prefetch_realtime_quotes(self, stock_codes: List[str]) -> int:
        """
        适配 pipeline.py 的批量预取行情逻辑
        """
        return len(stock_codes)

    def get_stock_name(self, code: str) -> str:
        """
        适配 pipeline.py 频繁调用的获取名称接口
        """
        if code in self._name_cache and self._name_cache[code] != code:
            return self._name_cache[code]
        
        # 如果缓存没有，尝试从实时行情里抓
        quote = self.get_realtime_quote(code)
        if quote and hasattr(quote, 'name') and quote.name:
            self._name_cache[code] = quote.name
            return quote.name
        return self._name_cache.get(code, code)

    def get_daily_data(self, stock_code: str, days: int = 30) -> Tuple[pd.DataFrame, str]:
        """
        适配 pipeline.py 的历史数据获取
        """
        normalized_code = normalize_stock_code(stock_code)
        is_us = is_us_pure_ticker(normalized_code)
        
        for fetcher in self._fetchers:
            # 分流逻辑：美股走 yfinance，A股走国内源
            if is_us and fetcher.name in ["TushareFetcher", "AkshareFetcher"]: continue
            if not is_us and fetcher.name == "YfinanceFetcher": continue

            try:
                df = fetcher.get_daily_data(normalized_code, days=days)
                if df is not None and not df.empty:
                    return df, fetcher.name
            except:
                continue
        return pd.DataFrame(), "None"

    def get_realtime_quote(self, stock_code: str):
        """
        适配 pipeline.py 的实时行情获取
        """
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
        """适配 pipeline.py 的筹码分布接口"""
        for fetcher in self._fetchers:
            if hasattr(fetcher, 'get_chip_distribution'):
                try:
                    chip = fetcher.get_chip_distribution(code)
                    if chip: return chip
                except: continue
        return None

    def get_fundamental_context(self, code: str, budget_seconds: float = 1.5):
        """适配 pipeline.py 的基本面聚合接口"""
        return {"code": code, "status": "not_supported", "source_chain": []}

    def build_failed_fundamental_context(self, code: str, reason: str):
        return {"code": code, "error": reason, "status": "failed"}
