# -*- coding: utf-8 -*-
"""
===================================
数据源基类与管理器 - 强制分流版
===================================
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

logger = logging.getLogger(__name__)

# === 标准化列名定义 ===
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']

# === 异常类定义 (修复 ImportError) ===

class DataFetchError(Exception):
    """数据获取异常基类"""
    pass

class RateLimitError(DataFetchError):
    """API 速率限制异常"""
    pass

class DataSourceUnavailableError(DataFetchError):
    """数据源不可用异常"""
    pass

# === 市场判定工具函数 (修复 ImportError) ===

def normalize_stock_code(stock_code: str) -> str:
    """标准化代码，去掉 .SH/.SZ 等后缀"""
    code = str(stock_code).strip().upper()
    if '.' in code:
        base, suffix = code.rsplit('.', 1)
        if suffix in ('US', 'SH', 'SZ', 'SS', 'BJ', 'HK'):
            return base
    
    # 处理前缀形式如 SH600519
    if code.startswith(('SH', 'SZ', 'BJ')) and code[2:].isdigit():
        return code[2:]
        
    return code

def canonical_stock_code(code: str) -> str:
    """返回规范化的大写代码（用于主程序入口）"""
    return (code or "").strip().upper()

def is_us_pure_ticker(code: str) -> bool:
    """判定是否为美股纯字母代码 (如 AAPL, PM)"""
    return bool(re.match(r'^[A-Z]{1,5}$', code))

def is_bse_code(code: str) -> bool:
    """判定是否为北交所代码"""
    c = normalize_stock_code(code)
    return c.startswith(("8", "4", "92"))

def is_st_stock(name: str) -> bool:
    """判定是否为 ST 股"""
    return 'ST' in (name or "").upper()

def is_kc_cy_stock(code: str) -> bool:
    """判定是否为科创板/创业板"""
    c = normalize_stock_code(code)
    return c.startswith("688") or c.startswith("30")

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
            
            # 基础指标计算
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'pct_chg']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['close']).sort_values('date')
            
            # MA 计算
            df['ma5'] = df['close'].rolling(5, min_periods=1).mean().round(2)
            df['ma10'] = df['close'].rolling(10, min_periods=1).mean().round(2)
            df['ma20'] = df['close'].rolling(20, min_periods=1).mean().round(2)
            
            return df
        except Exception as e:
            logger.error(f"[{self.name}] 获取 {stock_code} 报错: {str(e)}")
            return pd.DataFrame()

# === 策略管理器 ===

class DataFetcherManager:
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        # 延迟加载，防止循环引用
        from .efinance_fetcher import EfinanceFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .pytdx_fetcher import PytdxFetcher
        from .baostock_fetcher import BaostockFetcher
        from .yfinance_fetcher import YfinanceFetcher

        self._fetchers = [
            YfinanceFetcher(),
            EfinanceFetcher(),
            AkshareFetcher(),
            TushareFetcher(),
            PytdxFetcher(),
            BaostockFetcher()
        ]

    def get_daily_data(self, stock_code: str, **kwargs) -> Tuple[pd.DataFrame, str]:
        normalized_code = normalize_stock_code(stock_code)
        is_us = is_us_pure_ticker(normalized_code)
        
        errors = []
        for fetcher in self._fetchers:
            # 核心分流：美股不调 A 股源
            if is_us and fetcher.name in ["TushareFetcher", "BaostockFetcher", "PytdxFetcher"]:
                continue
            
            # A股不调 Yfinance
            if not is_us and fetcher.name == "YfinanceFetcher":
                continue

            try:
                df = fetcher.get_daily_data(normalized_code, **kwargs)
                if df is not None and not df.empty:
                    return df, fetcher.name
            except Exception as e:
                errors.append(f"{fetcher.name}: {str(e)}")
        
        raise DataFetchError(f"{stock_code} 获取失败: {'; '.join(errors)}")

    def get_realtime_quote(self, stock_code: str):
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
