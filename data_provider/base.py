# -*- coding: utf-8 -*-
"""
===================================
数据源基类与管理器
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

# --- 全局常量 ---
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']

# --- 异常类 ---
class DataFetchError(Exception):
    """基础数据获取异常"""
    pass

class RateLimitError(DataFetchError):
    """频率限制异常"""
    pass

class DataSourceUnavailableError(DataFetchError):
    """数据源不可用"""
    pass

# --- 工具函数 ---

def normalize_stock_code(stock_code: str) -> str:
    """标准化代码：去掉后缀，统一大写"""
    if not stock_code:
        return ""
    code = str(stock_code).strip().upper()
    if '.' in code:
        code = code.split('.')[0]
    return code

def canonical_stock_code(code: str) -> str:
    """返回规范的代码格式（带大写）"""
    return (code or "").strip().upper()

def is_bse_code(code: str) -> bool:
    """是否为北交所股票"""
    c = normalize_stock_code(code)
    return c.startswith(('8', '4', '92'))

def is_st_stock(name: str) -> bool:
    """是否为ST股票"""
    if not name:
        return False
    return 'ST' in name.upper()

def is_kc_cy_stock(code: str) -> bool:
    """是否为科创板或创业板"""
    c = normalize_stock_code(code)
    # 创业板300, 301; 科创板688
    return c.startswith(('300', '301', '688'))

def _is_hk_market(code: str) -> bool:
    """简单判定是否为港股代码 (5位数字)"""
    c = normalize_stock_code(code)
    return len(c) <= 5 and c.isdigit()

# --- 基类定义 ---

class BaseFetcher(ABC):
    """数据抓取基类"""
    name: str = "BaseFetcher"
    priority: int = 10  # 越小优先级越高
    
    def __init__(self):
        self._lock = RLock()
        self._semaphore = BoundedSemaphore(5)  # 默认并发限制

    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从数据源抓取原始数据"""
        pass

    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """将原始数据转换为统一格式"""
        pass

    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """公共接口：获取并清洗数据"""
        with self._semaphore:
            try:
                df = self._fetch_raw_data(stock_code, start_date, end_date)
                if df is None or df.empty:
                    return pd.DataFrame()
                
                df = self._normalize_data(df, stock_code)
                
                # 统一确保列存在
                for col in STANDARD_COLUMNS:
                    if col not in df.columns:
                        df[col] = np.nan
                
                # 转换日期格式
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                return df[STANDARD_COLUMNS]
            except Exception as e:
                logger.error(f"[{self.name}] 获取 {stock_code} 失败: {e}")
                raise

# --- 管理器 ---

class DataFetcherManager:
    """数据源管理器：支持多源回退 (Fallback)"""
    
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        if fetchers:
            self._fetchers = sorted(fetchers, key=lambda x: x.priority)
        else:
            self._fetchers = self._init_default_fetchers()

    def _init_default_fetchers(self) -> List[BaseFetcher]:
        """按顺序初始化默认数据源"""
        fetchers = []
        try:
            from .efinance_fetcher import EfinanceFetcher
            fetchers.append(EfinanceFetcher())
            
            from .akshare_fetcher import AkshareFetcher
            fetchers.append(AkshareFetcher())
            
            from .tushare_fetcher import TushareFetcher
            fetchers.append(TushareFetcher())
        except ImportError as e:
            logger.warning(f"部分Fetcher初始化失败 (可能缺少依赖): {e}")
        
        return sorted(fetchers, key=lambda x: x.priority)

    def get_daily_data(self, stock_code: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, str]:
        """尝试所有可用的 Fetcher，直到成功"""
        last_error = None
        for fetcher in self._fetchers:
            try:
                df = fetcher.get_daily_data(stock_code, start_date, end_date)
                if not df.empty:
                    return df, fetcher.name
            except Exception as e:
                last_error = e
                logger.warning(f"数据源 {fetcher.name} 获取 {stock_code} 失败，尝试下一个...")
                continue
        
        raise DataFetchError(f"所有数据源均无法获取 {stock_code}. 最后错误: {last_error}")
