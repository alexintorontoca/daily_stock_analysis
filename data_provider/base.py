# -*- coding: utf-8 -*-

"""

===================================

数据源基类与管理器

===================================



设计模式：策略模式 (Strategy Pattern)

- BaseFetcher: 抽象基类，定义统一接口

- DataFetcherManager: 策略管理器，实现自动切换



防封禁策略：

1. 每个 Fetcher 内置流控逻辑

2. 失败自动切换到下一个数据源

3. 指数退避重试机制

"""



import logging

import random

import time

from threading import BoundedSemaphore, RLock, Thread

from abc import ABC, abstractmethod

from datetime import datetime

from typing import Callable, Optional, List, Tuple, Dict, Any



import pandas as pd

import numpy as np

from src.data.stock_mapping import STOCK_NAME_MAP, is_meaningful_stock_name

from .fundamental_adapter import AkshareFundamentalAdapter



# 配置日志

logger = logging.getLogger(__name__)





# === 标准化列名定义 ===

STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']





def unwrap_exception(exc: Exception) -> Exception:

    """

    Follow chained exceptions and return the deepest non-cyclic cause.

    """

    current = exc

    visited = set()



    while current is not None and id(current) not in visited:

        visited.add(id(current))

        next_exc = current.__cause__ or current.__context__

        if next_exc is None:

            break

        current = next_exc



    return current





def summarize_exception(exc: Exception) -> Tuple[str, str]:

    """

    Build a stable summary for logs while preserving the application-layer message.

    """

    root = unwrap_exception(exc)

    error_type = type(root).__name__

    message = str(exc).strip() or str(root).strip() or error_type

    return error_type, " ".join(message.split())





def normalize_stock_code(stock_code: str) -> str:

    """

    Normalize stock code by stripping exchange prefixes/suffixes.



    Accepted formats and their normalized results:

    - '600519'      -> '600519'   (already clean)

    - 'SH600519'    -> '600519'   (strip SH prefix)

    - 'SZ000001'    -> '000001'   (strip SZ prefix)

    - 'BJ920748'    -> '920748'   (strip BJ prefix, BSE)

    - 'sh600519'    -> '600519'   (case-insensitive)

    - '600519.SH'   -> '600519'   (strip .SH suffix)

    - '000001.SZ'   -> '000001'   (strip .SZ suffix)

    - '920748.BJ'   -> '920748'   (strip .BJ suffix, BSE)

    - 'HK00700'     -> 'HK00700'  (keep HK prefix for HK stocks)

    - '1810.HK'     -> 'HK01810'  (normalize HK suffix to canonical prefix form)

    - 'AAPL'        -> 'AAPL'     (keep US stock ticker as-is)



    This function is applied at the DataProviderManager layer so that

    all individual fetchers receive a clean 6-digit code (for A-shares/ETFs).

    """

    code = stock_code.strip()

    upper = code.upper()



    # Normalize HK prefix to a canonical 5-digit form (e.g. hk1810 -> HK01810)

    if upper.startswith('HK') and not upper.startswith('HK.'):

        candidate = upper[2:]

        if candidate.isdigit() and 1 <= len(candidate) <= 5:

            return f"HK{candidate.zfill(5)}"



    # Strip SH/SZ prefix (e.g. SH600519 -> 600519)

    if upper.startswith(('SH', 'SZ')) and not upper.startswith('SH.') and not upper.startswith('SZ.'):

        candidate = code[2:]

        # Only strip if the remainder looks like a valid numeric code

        if candidate.isdigit() and len(candidate) in (5, 6):

            return candidate



    # Strip BJ prefix (e.g. BJ920748 -> 920748)

    if upper.startswith('BJ') and not upper.startswith('BJ.'):

        candidate = code[2:]

        if candidate.isdigit() and len(candidate) == 6:

            return candidate



    # Strip .SH/.SZ/.BJ suffix (e.g. 600519.SH -> 600519, 920748.BJ -> 920748)

    if '.' in code:

        base, suffix = code.rsplit('.', 1)

        if suffix.upper() == 'HK' and base.isdigit() and 1 <= len(base) <= 5:

            return f"HK{base.zfill(5)}"

        if suffix.upper() in ('SH', 'SZ', 'SS', 'BJ') and base.isdigit():

            return base



    return code





ETF_PREFIXES = ("51", "52", "56", "58", "15", "16", "18")





def _is_us_market(code: str) -> bool:

    """判断是否为美股/美股指数代码（不含中文前后缀）。"""

    from .us_index_mapping import is_us_stock_code, is_us_index_code



    normalized = (code or "").strip().upper()

    return is_us_index_code(normalized) or is_us_stock_code(normalized)





def _is_hk_market(code: str) -> bool:

    """

    判定是否为港股代码。



    支持 `HK00700` 及纯 5 位数字形式（A 股 ETF/股票常见为 6 位）。

    """

    normalized = (code or "").strip().upper()

    if normalized.endswith(".HK"):

        base = normalized[:-3]

        return base.isdigit() and 1 <= len(base) <= 5

    if normalized.startswith("HK"):

        digits = normalized[2:]

        return digits.isdigit() and 1 <= len(digits) <= 5

    if normalized.isdigit() and len(normalized) == 5:

        return True

    return False





def _is_etf_code(code: str) -> bool:

    """判定 A 股 ETF 基金代码（保守规则）。"""

    normalized = normalize_stock_code(code)

    return (

        normalized.isdigit()

        and len(normalized) == 6

        and normalized.startswith(ETF_PREFIXES)

    )





def _market_tag(code: str) -> str:

    """返回市场标签: cn/us/hk."""

    if _is_us_market(code):

        return "us"

    if _is_hk_market(code):

        return "hk"

    return "cn"





def is_bse_code(code: str) -> bool:

    """

    Check if the code is a Beijing Stock Exchange (BSE) A-share code.



    BSE rules:

    - Old format (pre-2024): 8xxxxx (e.g. 838163), 4xxxxx (e.g. 430047)

    - New format (2024+, post full migration Oct 2025): 920xxx+

    Note: 900xxx are Shanghai B-shares, NOT BSE — must return False.

    """

    c = (code or "").strip().split(".")[0]

    if len(c) != 6 or not c.isdigit():

        return False

    return c.startswith(("8", "4")) or c.startswith("92")



def is_st_stock(name: str) -> bool:

    """

    Check if the stock is an ST or *ST stock based on its name.



    ST stocks have special trading rules and typically a ±5% limit.

    """

    n = (name or "").upper()

    return 'ST' in n



def is_kc_cy_stock(code: str) -> bool:

    """

    Check if the stock is a STAR Market (科创板) or ChiNext (创业板) stock based on its code.



    - STAR Market: Codes starting with 688

    - ChiNext: Codes starting with 300

    Both have a ±20% limit.

    """

    c = (code or "").strip().split(".")[0]

    return c.startswith("688") or c.startswith("30")





def canonical_stock_code(code: str) -> str:

    """

    Return the canonical (uppercase) form of a stock code.



    This is a display/storage layer concern, distinct from normalize_stock_code

    which strips exchange prefixes. Apply at system input boundaries to ensure

    consistent case across BOT, WEB UI, API, and CLI paths (Issue #355).



    Examples:

        'aapl'    -> 'AAPL'

        'AAPL'    -> 'AAPL'

        '600519'  -> '600519'  (digits are unchanged)

        'hk00700' -> 'HK00700'

    """

    return (code or "").strip().upper()





class DataFetchError(Exception):

    """数据获取异常基类"""

    pass





class RateLimitError(DataFetchError):

    """API 速率限制异常"""

    pass





class DataSourceUnavailableError(DataFetchError):

    """数据源不可用异常"""

    pass





class BaseFetcher(ABC):

    """

    数据源抽象基类

    

    职责：

    1. 定义统一的数据获取接口

    2. 提供数据标准化方法

    3. 实现通用的技术指标计算

    

    子类实现：

    - _fetch_raw_data(): 从具体数据源获取原始数据

    - _normalize_data(): 将原始数据转换为标准格式

    """

    

    name: str = "BaseFetcher"

    priority: int = 99  # 优先级数字越小越优先

    

    @abstractmethod

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:

        """

        从数据源获取原始数据（子类必须实现）

        

        Args:

            stock_code: 股票代码，如 '600519', '000001'

            start_date: 开始日期，格式 'YYYY-MM-DD'

            end_date: 结束日期，格式 'YYYY-MM-DD'

            

        Returns:

            原始数据 DataFrame（列名因数据源而异）

        """

        pass

    

    @abstractmethod

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:

        """

        标准化数据列名（子类必须实现）



        将不同数据源的列名统一为：

        ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']

        """

        pass



    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:

        """

        获取主要指数实时行情



        Args:

            region: 市场区域，cn=A股 us=美股



        Returns:

            List[Dict]: 指数列表，每个元素为字典，包含:

                - code: 指数代码

                - name: 指数名称

                - current: 当前点位

                - change: 涨跌点数

                - change_pct: 涨跌幅(%)

                - volume: 成交量

                - amount: 成交额

        """

        return None



    def get_market_stats(self) -> Optional[Dict[str, Any]]:

        """

        获取市场涨跌统计



        Returns:

            Dict: 包含:

                - up_count: 上涨家数

                - down_count: 下跌家数

                - flat_count: 平盘家数

                - limit_up_count: 涨停家数

                - limit_down_count: 跌停家数

                - total_amount: 两市成交额

        """

        return None



    def get_sector_rankings(self, n: int = 5) -> Optional[Tuple[List[Dict], List[Dict]]]:

        """

        获取板块涨跌榜



        Args:

            n: 返回前n个



        Returns:

            Tuple: (领涨板块列表, 领跌板块列表)

        """

        return None



    def get_daily_data(

        self,

        stock_code: str, 

        start_date: Optional[str] = None,

        end_date: Optional[str] = None,

        days: int = 30

    ) -> pd.DataFrame:

        """

        获取日线数据（统一入口）

        

        流程：

        1. 计算日期范围

        2. 调用子类获取原始数据

        3. 标准化列名

        4. 计算技术指标

        

        Args:

            stock_code: 股票代码

            start_date: 开始日期（可选）

            end_date: 结束日期（可选，默认今天）

            days: 获取天数（当 start_date 未指定时使用）

            

        Returns:

            标准化的 DataFrame，包含技术指标

        """

        # 计算日期范围

        if end_date is None:

            end_date = datetime.now().strftime('%Y-%m-%d')

        

        if start_date is None:

            # 默认获取最近 30 个交易日（按日历日估算，多取一些）

            from datetime import timedelta

            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)

            start_date = start_dt.strftime('%Y-%m-%d')



        request_start = time.time()

        logger.info(f"[{self.name}] 开始获取 {stock_code} 日线数据: 范围={start_date} ~ {end_date}")

        

        try:

            # Step 1: 获取原始数据

            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)

            

            if raw_df is None or raw_df.empty:

                raise DataFetchError(f"[{self.name}] 未获取到 {stock_code} 的数据")

            

            # Step 2: 标准化列名

            df = self._normalize_data(raw_df, stock_code)

            

            # Step 3: 数据清洗

            df = self._clean_data(df)

            

            # Step 4: 计算技术指标

            df = self._calculate_indicators(df)



            elapsed = time.time() - request_start

            logger.info(

                f"[{self.name}] {stock_code} 获取成功: 范围={start_date} ~ {end_date}, "

                f"rows={len(df)}, elapsed={elapsed:.2f}s"

            )

            return df

            

        except Exception as e:

            elapsed = time.time() - request_start

            error_type, error_reason = summarize_exception(e)

            logger.error(

                f"[{self.name}] {stock_code} 获取失败: 范围={start_date} ~ {end_date}, "

                f"error_type={error_type}, elapsed={elapsed:.2f}s, reason={error_reason}"

            )

            raise DataFetchError(f"[{self.name}] {stock_code}: {error_reason}") from e

    

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:

        """

        数据清洗

        

        处理：

        1. 确保日期列格式正确

        2. 数值类型转换

        3. 去除空值行

        4. 按日期排序

        """

        df = df.copy()

        

        # 确保日期列为 datetime 类型

        if 'date' in df.columns:

            df['date'] = pd.to_datetime(df['date'])

        

        # 数值列类型转换

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']

        for col in numeric_cols:

            if col in df.columns:

                df[col] = pd.to_numeric(df[col], errors='coerce')

        

        # 去除关键列为空的行

        df = df.dropna(subset=['close', 'volume'])

        

        # 按日期升序排序

        df = df.sort_values('date', ascending=True).reset_index(drop=True)

        

        return df

    

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:

        """

        计算技术指标

        

        计算指标：

        - MA5, MA10, MA20: 移动平均线

        - Volume_Ratio: 量比（今日成交量 / 5日平均成交量）

        """

        df = df.copy()

        

        # 移动平均线

        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()

        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()

        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()

        

        # 量比：当日成交量 / 5日平均成交量

        # 注意：此处的 volume_ratio 是“日线成交量 / 前5日均量(shift 1)”的相对倍数，

        # 与部分交易软件口径的“分时量比（同一时刻对比）”不同，含义更接近“放量倍数”。

        # 该行为目前保留（按需求不改逻辑）。

        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()

        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)

        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

        

        # 保留2位小数

        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio']:

            if col in df.columns:

                df[col] = df[col].round(2)

        

        return df

    

    @staticmethod

    def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:

        """

        智能随机休眠（Jitter）

        

        防封禁策略：模拟人类行为的随机延迟

        在请求之间加入不规则的等待时间

        """

        sleep_time = random.uniform(min_seconds, max_seconds)

        logger.debug(f"随机休眠 {sleep_time:.2f} 秒...")

        time.sleep(sleep_time)





class DataFetcherManager:

    """

    数据源策略管理器

    

    职责：

    1. 管理多个数据源（按优先级排序）

    2. 自动故障切换（Failover）

    3. 提供统一的数据获取接口

    

    切换策略：

    - 优先使用高优先级数据源

    - 失败后自动切换到下一个

    - 所有数据源都失败时抛出异常

    """

    

    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):

        """

        初始化管理器

        

        Args:

            fetchers: 数据源列表（可选，默认按优先级自动创建）

        """

        self._fetchers: List[BaseFetcher] = []

        

        if fetchers:

            # 按优先级排序

            self._fetchers = sorted(fetchers, key=lambda f: f.priority)

        else:

            # 默认数据源将在首次使用时延迟加载

            self._init_default_fetchers()

        self._fundamental_adapter = AkshareFundamentalAdapter()

        self._fundamental_cache: Dict[str, Dict[str, Any]] = {}

        self._fundamental_cache_lock = RLock()

        self._fundamental_timeout_worker_limit = 8

        self._fundamental_timeout_slots = BoundedSemaphore(self._fundamental_timeout_worker_limit)



    def _get_fundamental_cache_key(self, stock_code: str, budget_seconds: Optional[float] = None) -> str:

        """生成基本面缓存 key（包含预算分桶以避免低预算结果污染高预算请求）。"""

        normalized_code = normalize_stock_code(stock_code)

        if budget_seconds is None:

            return f"{normalized_code}|budget=default"

        try:

            budget = max(0.0, float(budget_seconds))

        except (TypeError, ValueError):

            budget = 0.0

        # 100ms bucket to balance cache reuse and scenario isolation.

        budget_bucket = int(round(budget * 10))

        return f"{normalized_code}|budget={budget_bucket}"



    def _prune_fundamental_cache(self, ttl_seconds: int, max_entries: int) -> None:

        """Prune expired and overflow fundamental cache items."""

        with self._fundamental_cache_lock:

            if not self._fundamental_cache:

                return



            now_ts = time.time()

            if ttl_seconds > 0:

                cache_items = list(self._fundamental_cache.items())

                expired_keys = [

                    key

                    for key, value in cache_items

                    if now_ts - float(value.get("ts", 0)) > ttl_seconds

                ]

                for key in expired_keys:

                    self._fundamental_cache.pop(key, None)



            if max_entries > 0 and len(self._fundamental_cache) > max_entries:

                overflow = len(self._fundamental_cache) - max_entries

                sorted_items = sorted(

                    list(self._fundamental_cache.items()),

                    key=lambda item: float(item[1].get("ts", 0)),

                )

                for key, _ in sorted_items[:overflow]:

                    self._fundamental_cache.pop(key, None)



    @staticmethod

    def _is_missing_board_value(value: Any) -> bool:

        """Return True when a board field value should be treated as missing."""

        if value is None:

            return True

        try:

            if pd.isna(value):

                return True

        except Exception:

            pass

        text = str(value).strip()

        return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}



    @staticmethod

    def _normalize_belong_boards(raw_data: Any) -> List[Dict[str, Any]]:

        """Normalize belong-board results from heterogeneous providers."""

        if DataFetcherManager._is_missing_board_value(raw_data):

            return []



        normalized: List[Dict[str, Any]] = []

        dedupe = set()



        if isinstance(raw_data, pd.DataFrame):

            if raw_data.empty:

                return []

            name_col = next(

                (

                    col

                    for col in raw_data.columns

                    if str(col) in {"板块名称", "板块", "所属板块", "板块名", "name", "industry"}

                ),

                None,

            )

            code_col = next(

                (

                    col

                    for col in raw_data.columns

                    if str(col) in {"板块代码", "代码", "code"}

                ),

                None,

            )

            type_col = next(

                (

                    col

                    for col in raw_data.columns

                    if str(col) in {"板块类型", "类别", "type"}

                ),

                None,

            )

            if name_col is None:

                return []

            for _, row in raw_data.iterrows():

                board_name_raw = row.get(name_col, "")

                if DataFetcherManager._is_missing_board_value(board_name_raw):

                    continue

                board_name = str(board_name_raw).strip()

                if board_name in dedupe:

                    continue

                dedupe.add(board_name)

                item = {"name": board_name}

                if code_col is not None:

                    board_code_raw = row.get(code_col, "")

                    if not DataFetcherManager._is_missing_board_value(board_code_raw):

                        item["code"] = str(board_code_raw).strip()

                if type_col is not None:

                    board_type_raw = row.get(type_col, "")

                    if not DataFetcherManager._is_missing_board_value(board_type_raw):

                        item["type"] = str(board_type_raw).strip()

                normalized.append(item)

            return normalized



        if isinstance(raw_data, dict):

            raw_data = [raw_data]



        if isinstance(raw_data, (list, tuple, set)):

            for item in raw_data:

                if isinstance(item, dict):

                    board_name_raw = (

                        item.get("name")

                        or item.get("board_name")

                        or item.get("板块名称")

                        or item.get("板块")

                        or item.get("所属板块")

                        or item.get("板块名")

                        or item.get("industry")

                        or item.get("行业")

                    )

                    if DataFetcherManager._is_missing_board_value(board_name_raw):

                        continue

                    board_name = str(board_name_raw).strip()

                    if board_name in dedupe:

                        continue

                    dedupe.add(board_name)

                    normalized_item: Dict[str, Any] = {"name": board_name}

                    code_raw = (

                        item.get("code")

                        or item.get("板块代码")

                        or item.get("代码")

                    )

                    if not DataFetcherManager._is_missing_board_value(code_raw):

                        normalized_item["code"] = str(code_raw).strip()

                    type_raw = (

                        item.get("type")

                        or item.get("板块类型")

                        or item.get("类别")

                    )

                    if not DataFetcherManager._is_missing_board_value(type_raw):

                        normalized_item["type"] = str(type_raw).strip()

                    normalized.append(normalized_item)

                    continue

                if DataFetcherManager._is_missing_board_value(item):

                    continue

                board_name = str(item).strip()

                if board_name in dedupe:

                    continue

                dedupe.add(board_name)

                normalized.append({"name": board_name})

            return normalized



        if not DataFetcherManager._is_missing_board_value(raw_data):

            board_name = str(raw_data).strip()

            return [{"name": board_name}]

        return []

    

    def _init_default_fetchers(self) -> None:

        """

        初始化默认数据源列表



        优先级动态调整逻辑：

        - 如果配置了 TUSHARE_TOKEN：Tushare 优先级提升为 0（最高）

        - 否则按默认优先级：

          0. EfinanceFetcher (Priority 0) - 最高优先级

          1. AkshareFetcher (Priority 1)

          2. PytdxFetcher (Priority 2) - 通达信

          2. TushareFetcher (Priority 2)

          3. BaostockFetcher (Priority 3)

          4. YfinanceFetcher (Priority 4)

        """

        from .efinance_fetcher import EfinanceFetcher

        from .akshare_fetcher import AkshareFetcher

        from .tushare_fetcher import TushareFetcher

        from .pytdx_fetcher import PytdxFetcher

        from .baostock_fetcher import BaostockFetcher

        from .yfinance_fetcher import YfinanceFetcher

        # 创建所有数据源实例（优先级在各 Fetcher 的 __init__ 中确定）

        efinance = EfinanceFetcher()

        akshare = AkshareFetcher()

        tushare = TushareFetcher()  # 会根据 Token 配置自动调整优先级

        pytdx = PytdxFetcher()      # 通达信数据源（可配 PYTDX_HOST/PYTDX_PORT）

        baostock = BaostockFetcher()

        yfinance = YfinanceFetcher()



        # 初始化数据源列表

        self._fetchers = [

            efinance,

            akshare,

            tushare,

            pytdx,

            baostock,

            yfinance,

        ]



        # 按优先级排序（Tushare 如果配置了 Token 且初始化成功，优先级为 0）

        self._fetchers.sort(key=lambda f: f.priority)



        # 构建优先级说明

        priority_info = ", ".join([f"{f.name}(P{f.priority})" for f in self._fetchers])

        logger.info(f"已初始化 {len(self._fetchers)} 个数据源（按优先级）: {priority_info}")

    

    def add_fetcher(self, fetcher: BaseFetcher) -> None:

        """添加数据源并重新排序"""

        self._fetchers.append(fetcher)

        self._fetchers.sort(key=lambda f: f.priority)

    

    def get_daily_data(

        self, 

        stock_code: str,

        start_date: Optional[str] = None,

        end_date: Optional[str] = None,

        days: int = 30

    ) -> Tuple[pd.DataFrame, str]:

        """

        获取日线数据（自动切换数据源）

        

        故障切换策略：

        1. 美股指数/美股股票直接路由到 YfinanceFetcher

        2. 其他代码从最高优先级数据源开始尝试

        3. 捕获异常后自动切换到下一个

        4. 记录每个数据源的失败原因

        5. 所有数据源失败后抛出详细异常

        

        Args:

            stock_code: 股票代码

            start_date: 开始日期

            end_date: 结束日期

            days: 获取天数

            

        Returns:

            Tuple[DataFrame, str]: (数据, 成功的数据源名称)

            

        Raises:

            DataFetchError: 所有数据源都失败时抛出

        """

        from .us_index_mapping import is_us_index_code, is_us_stock_code



        # Normalize code (strip SH/SZ prefix etc.)

        stock_code = normalize_stock_code(stock_code)



        errors = []

        total_fetchers = len(self._fetchers)

        request_start = time.time()



        # 快速路径：美股指数与美股股票直接路由到 YfinanceFetcher

        if is_us_index_code(stock_code) or is_us_stock_code(stock_code):

            for attempt, fetcher in enumerate(self._fetchers, start=1):

                if fetcher.name == "YfinanceFetcher":

                    try:

                        logger.info(

                            f"[数据源尝试 {attempt}/{total_fetchers}] [{fetcher.name}] "

                            f"美股/美股指数 {stock_code} 直接路由..."

                        )

                        df = fetcher.get_daily_data(

                            stock_code=stock_code,

                            start_date=start_date,

                            end_date=end_date,

                            days=days,

                        )

                        if df is not None and not df.empty:

                            elapsed = time.time() - request_start

                            logger.info(

                                f"[数据源完成] {stock_code} 使用 [{fetcher.name}] 获取成功: "

                                f"rows={len(df)}, elapsed={elapsed:.2f}s"

                            )

                            return df, fetcher.name

                    except Exception as e:

                        error_type, error_reason = summarize_exception(e)

                        error_msg = f"[{fetcher.name}] ({error_type}) {error_reason}"

                        logger.warning(

                            f"[数据源失败 {attempt}/{total_fetchers}] [{fetcher.name}] {stock_code}: "

                            f"error_type={error_type}, reason={error_reason}"

                        )

                        errors.append(error_msg)

                    break

            # YfinanceFetcher failed or not found

            error_summary = f"美股/美股指数 {stock_code} 获取失败:\n" + "\n".join(errors)

            elapsed = time.time() - request_start

            logger.error(f"[数据源终止] {stock_code} 获取失败: elapsed={elapsed:.2f}s\n{error_summary}")

            raise DataFetchError(error_summary)



        for attempt, fetcher in enumerate(self._fetchers, start=1):

            try:

                logger.info(f"[数据源尝试 {attempt}/{total_fetchers}] [{fetcher.name}] 获取 {stock_code}...")

                df = fetcher.get_daily_data(

                    stock_code=stock_code,

                    start_date=start_date,

                    end_date=end_date,

                    days=days

                )

                

                if df is not None and not df.empty:

                    elapsed = time.time() - request_start

                    logger.info(

                        f"[数据源完成] {stock_code} 使用 [{fetcher.name}] 获取成功: "

                        f"rows={len(df)}, elapsed={elapsed:.2f}s"

                    )

                    return df, fetcher.name

                    

            except Exception as e:

                error_type, error_reason = summarize_exception(e)

                error_msg = f"[{fetcher.name}] ({error_type}) {error_reason}"

                logger.warning(

                    f"[数据源失败 {attempt}/{total_fetchers}] [{fetcher.name}] {stock_code}: "

                    f"error_type={error_type}, reason={error_reason}"

                )

                errors.append(error_msg)

                if attempt < total_fetchers:

                    next_fetcher = self._fetchers[attempt]

                    logger.info(f"[数据源切换] {stock_code}: [{fetcher.name}] -> [{next_fetcher.name}]")

                # 继续尝试下一个数据源

                continue

        

        # 所有数据源都失败

        error_summary = f"所有数据源获取 {stock_code} 失败:\n" + "\n".join(errors)

        elapsed = time.time() - request_start

        logger.error(f"[数据源终止] {stock_code} 获取失败: elapsed={elapsed:.2f}s\n{error_summary}")

        raise DataFetchError(error_summary)

    

    @property

    def available_fetchers(self) -> List[str]:

        """返回可用数据源名称列表"""

        return [f.name for f in self._fetchers]

    

    def prefetch_realtime_quotes(self, stock_codes: List[str]) -> int:

        """

        批量预取实时行情数据（在分析开始前调用）

        

        策略：

        1. 检查优先级中是否包含全量拉取数据源（efinance/akshare_em）

        2. 如果不包含，跳过预取（新浪/腾讯是单股票查询，无需预取）

        3. 如果自选股数量 >= 5 且使用全量数据源，则预取填充缓存

        

        这样做的好处：

        - 使用新浪/腾讯时：每只股票独立查询，无全量拉取问题

        - 使用 efinance/东财时：预取一次，后续缓存命中

        

        Args:

            stock_codes: 待分析的股票代码列表

            

        Returns:

            预取的股票数量（0 表示跳过预取）

        """

        # Normalize all codes

        stock_codes = [normalize_stock_code(c) for c in stock_codes]



        from src.config import get_config



        config = get_config()



        # Issue #455: PREFETCH_REALTIME_QUOTES=false 可禁用预取，避免全市场拉取

        if not getattr(config, "prefetch_realtime_quotes", True):

            logger.debug("[预取] PREFETCH_REALTIME_QUOTES=false，跳过批量预取")

            return 0



        # 如果实时行情被禁用，跳过预取

        if not config.enable_realtime_quote:

            logger.debug("[预取] 实时行情功能已禁用，跳过预取")

            return 0

        

        # 检查优先级中是否包含全量拉取数据源

        # 注意：新增全量接口（如 tushare_realtime）时需同步更新此列表

        # 全量接口特征：一次 API 调用拉取全市场 5000+ 股票数据

        priority = config.realtime_source_priority.lower()

        bulk_sources = ['efinance', 'akshare_em', 'tushare']  # 全量接口列表

        

        # 如果优先级中前两个都不是全量数据源，跳过预取

        # 因为新浪/腾讯是单股票查询，不需要预取

        priority_list = [s.strip() for s in priority.split(',')]

        first_bulk_source_index = None

        for i, source in enumerate(priority_list):

            if source in bulk_sources:

                first_bulk_source_index = i

                break

        

        # 如果没有全量数据源，或者全量数据源排在第 3 位之后，跳过预取

        if first_bulk_source_index is None or first_bulk_source_index >= 2:

            logger.info(f"[预取] 当前优先级使用轻量级数据源(sina/tencent)，无需预取")

            return 0

        

        # 如果股票数量少于 5 个，不进行批量预取（逐个查询更高效）

        if len(stock_codes) < 5:

            logger.info(f"[预取] 股票数量 {len(stock_codes)} < 5，跳过批量预取")

            return 0

        

        logger.info(f"[预取] 开始批量预取实时行情，共 {len(stock_codes)} 只股票...")

        

        # 尝试通过 efinance 或 akshare 预取

        # 只需要调用一次 get_realtime_quote，缓存机制会自动拉取全市场数据

        try:

            # 用第一只股票触发全量拉取

            first_code = stock_codes[0]

            quote = self.get_realtime_quote(first_code)

            

            if quote:

                logger.info(f"[预取] 批量预取完成，缓存已填充")

                return len(stock_codes)

            else:

                logger.warning(f"[预取] 批量预取失败，将使用逐个查询模式")

                return 0

                

        except Exception as e:

            logger.error(f"[预取] 批量预取异常: {e}")

            return 0

    

    def get_realtime_quote(self, stock_code: str):

        """

        获取实时行情数据（自动故障切换）

        

        故障切换策略（按配置的优先级）：

        1. 美股：使用 YfinanceFetcher.get_realtime_quote()

        2. EfinanceFetcher.get_realtime_quote()

        3. AkshareFetcher.get_realtime_quote(source="em")  - 东财

        4. AkshareFetcher.get_realtime_quote(source="sina") - 新浪

        5. AkshareFetcher.get_realtime_quote(source="tencent") - 腾讯

        6. 返回 None（降级兜底）

        

        Args:

            stock_code: 股票代码

            

        Returns:

            UnifiedRealtimeQuote 对象，所有数据源都失败则返回 None

        """

        # Normalize code (strip SH/SZ prefix etc.)

        stock_code = normalize_stock_code(stock_code)



        from .akshare_fetcher import _is_us_code

        from .us_index_mapping import is_us_index_code

        from src.config import get_config



        config = get_config()



        # 如果实时行情功能被禁用，直接返回 None

        if not config.enable_realtime_quote:

            logger.debug(f"[实时行情] 功能已禁用，跳过 {stock_code}")

            return None



        # 美股指数由 YfinanceFetcher 处理（在美股股票检查之前）

        if is_us_index_code(stock_code):

            for fetcher in self._fetchers:

                if fetcher.name == "YfinanceFetcher":

                    if hasattr(fetcher, 'get_realtime_quote'):

                        try:

                            quote = fetcher.get_realtime_quote(stock_code)

                            if quote is not None:

                                logger.info
