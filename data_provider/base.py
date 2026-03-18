# -*- coding: utf-8 -*-
"""
===================================
数据源基类与管理器 - 强制分流版
===================================
核心改动：
1. 彻底拦截：只要股票代码是纯字母（美股），严禁调用 Tushare/Baostock/Pytdx。
2. 自动纠错：不再需要外部环境变量，代码内部自动识别市场并分配最强 Fetcher。
"""

import logging
import re
import random
import time
from threading import BoundedSemaphore, RLock
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd
import numpy as np
from .fundamental_adapter import AkshareFundamentalAdapter

logger = logging.getLogger(__name__)

# === 基础工具函数 ===

def normalize_stock_code(stock_code: str) -> str:
    """标准化代码，去掉 .SH/.SZ 等后缀"""
    code = str(stock_code).strip().upper()
    if '.' in code:
        base, suffix = code.rsplit('.', 1)
        # 如果后缀是 US 或者是纯字母（美股），返回 base
        if suffix in ('US', 'SH', 'SZ', 'SS', 'BJ'):
            return base
    return code

def is_us_pure_ticker(code: str) -> bool:
    """判定是否为美股纯字母代码 (如 AAPL, PM)"""
    # 匹配 1-5 位纯大写字母
    return bool(re.match(r'^[A-Z]{1,5}$', code))

class DataFetchError(Exception):
    pass

# === Fetcher 基类 ===

class BaseFetcher(ABC):
    name: str = "BaseFetcher"
    priority: int = 99
    
    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        pass

    def get_daily_data(self, stock_code: str, start_date: Optional[str] = None, end_date: Optional[str] = None, days: int = 30) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            from datetime import timedelta
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')

        try:
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)
            if raw_df is None or raw_df.empty:
                return pd.DataFrame()
            df = self._normalize_data(raw_df, stock_code)
            # 数据清洗与指标计算...
            return df
        except Exception as e:
            raise DataFetchError(f"[{self.name}] 获取 {stock_code} 失败: {str(e)}")

# === 策略管理器 (核心分流逻辑所在) ===

class DataFetcherManager:
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        from .efinance_fetcher import EfinanceFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .pytdx_fetcher import PytdxFetcher
        from .baostock_fetcher import BaostockFetcher
        from .yfinance_fetcher import YfinanceFetcher

        # 这里的顺序就是默认优先级
        self._fetchers = [
            YfinanceFetcher(),  # 美股之王，排第一
            EfinanceFetcher(),  # A股/港股 东方财富接口，备选之王
            AkshareFetcher(),
            TushareFetcher(),   # 只有 A 股且 Efinance 挂了才会轮到它
            PytdxFetcher(),
            BaostockFetcher()
        ]

    def get_daily_data(self, stock_code: str, **kwargs) -> Tuple[pd.DataFrame, str]:
        normalized_code = normalize_stock_code(stock_code)
        is_us = is_us_pure_ticker(normalized_code)
        
        errors = []
        for fetcher in self._fetchers:
            # --- 核心拦截：美股代码绝对不走国内数据源 ---
            if is_us and fetcher.name in ["TushareFetcher", "BaostockFetcher", "PytdxFetcher"]:
                logger.debug(f"跳过国内源 {fetcher.name} 处理美股 {normalized_code}")
                continue

            # --- 核心拦截：A股数字代码（如 600519）没必要走 Yfinance ---
            if not is_us and fetcher.name == "YfinanceFetcher":
                continue

            try:
                df = fetcher.get_daily_data(normalized_code, **kwargs)
                if df is not None and not df.empty:
                    return df, fetcher.name
            except Exception as e:
                errors.append(f"{fetcher.name}: {str(e)}")
        
        raise DataFetchError(f"{stock_code} 所有数据源均失效: {'; '.join(errors)}")

    def get_realtime_quote(self, stock_code: str):
        """获取实时行情，应用相同的拦截逻辑"""
        normalized_code = normalize_stock_code(stock_code)
        is_us = is_us_pure_ticker(normalized_code)

        for fetcher in self._fetchers:
            if is_us and fetcher.name in ["TushareFetcher", "BaostockFetcher", "PytdxFetcher"]:
                continue
            if not is_us and fetcher.name == "YfinanceFetcher":
                continue
                
            if hasattr(fetcher, 'get_realtime_quote'):
                try:
                    quote = fetcher.get_realtime_quote(normalized_code)
                    if quote: return quote
                except: continue
        return None
