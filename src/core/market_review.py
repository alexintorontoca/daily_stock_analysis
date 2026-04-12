# -*- coding: utf-8 -*-
"""
===================================
股票智能分析系统 - 大盘复盘模块（支持 A 股 / 美股）
===================================

职责：
1. 根据 MARKET_REVIEW_REGION 配置选择市场区域（cn / us / both）
2. 执行大盘复盘分析并生成复盘报告
3. 保存和发送复盘报告
"""

import logging
from datetime import datetime
from typing import Optional
import pytz  # 引入时区处理

from src.config import get_config
from src.notification import NotificationService
from src.market_analyzer import MarketAnalyzer
from src.report_language import normalize_report_language
from src.search_service import SearchService
from src.analyzer import GeminiAnalyzer


logger = logging.getLogger(__name__)


def _get_market_review_text(language: str) -> dict[str, str]:
    normalized = normalize_report_language(language)
    if normalized == "en":
        return {
            "root_title": "# 🎯 Market Review",
            "push_title": "🎯 Market Review",
            "cn_title": "# A-share Market Recap",
            "us_title": "# US Market Recap",
            "separator": "> US market recap follows",
        }
    return {
        "root_title": "# 🎯 大盘复盘",
        "push_title": "🎯 大盘复盘",
        "cn_title": "# A股大盘复盘",
        "us_title": "# 美股大盘复盘",
        "separator": "> 以下为美股大盘复盘",
    }


def run_market_review(
    notifier: NotificationService,
    analyzer: Optional[GeminiAnalyzer] = None,
    search_service: Optional[SearchService] = None,
    send_notification: bool = True,
    merge_notification: bool = False,
    override_region: Optional[str] = None,
) -> Optional[str]:
    """
    执行大盘复盘分析

    Args:
        notifier: 通知服务
        analyzer: AI分析器（可选）
        search_service: 搜索服务（可选）
        send_notification: 是否发送通知
        merge_notification: 是否合并推送（跳过本次推送，由 main 层合并个股+大盘后统一发送，Issue #190）
        override_region: 覆盖 config 的 market_review_region（Issue #373 交易日过滤后有效子集）

    Returns:
        复盘报告文本
    """
    logger.info("开始执行大盘复盘分析...")
    config = get_config()
    review_text = _get_market_review_text(getattr(config, "report_language", "zh"))
    
    # === 新增逻辑：基于北京时间自动判定市场类型 ===
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now_bj = datetime.now(beijing_tz)
    hour_bj = now_bj.hour
    
    # 默认从配置获取 region
    region = (
        override_region
        if override_region is not None
        else (getattr(config, 'market_review_region', 'cn') or 'cn')
    )

    # 如果配置是 'both'，我们根据当前时间进行自动切分
    # A股收盘后(15:00-19:00)执行CN，美股收盘后(04:00-10:00)执行US
    if region == 'both':
        if 13 <= hour_bj <= 19:
            auto_region = 'cn'
            logger.info(f"当前北京时间 {hour_bj}点，自动切换至 A 股复盘模式")
        elif 4 <= hour_bj <= 11:
            auto_region = 'us'
            logger.info(f"当前北京时间 {hour_bj}点，自动切换至美股复盘模式")
        else:
            auto_region = 'both' # 其他时间保持原样
    else:
        auto_region = region

    if auto_region not in ('cn', 'us', 'both'):
        auto_region = 'cn'
    # ============================================

    try:
        if auto_region == 'both':
            # 顺序执行 A 股 + 美股，合并报告
            cn_analyzer = MarketAnalyzer(
                search_service=search_service, analyzer=analyzer, region='cn'
            )
            us_analyzer = MarketAnalyzer(
                search_service=search_service, analyzer=analyzer, region='us'
            )
            logger.info("生成 A 股大盘复盘报告...")
            cn_report = cn_analyzer.run_daily_review()
            logger.info("生成美股大盘复盘报告...")
            us_report = us_analyzer.run_daily_review()
            review_report = ''
            if cn_report:
                review_report = f"{review_text['cn_title']}\n\n{cn_report}"
            if us_report:
                if review_report:
                    review_report += f"\n\n---\n\n{review_text['separator']}\n\n"
                review_report += f"{review_text['us_title']}\n\n{us_report}"
            if not review_report:
                review_report = None
        else:
            market_analyzer = MarketAnalyzer(
                search_service=search_service,
                analyzer=analyzer,
                region=auto_region, # 使用判定后的 auto_region
            )
            review_report = market_analyzer.run_daily_review()
        
        if review_report:
            # 保存报告到文件
            date_str = now_bj.strftime('%Y%m%d')
            # === 修改点：文件名加上市场后缀以便区分 ===
            market_suffix = auto_region.upper()
            report_filename = f"market_review_{market_suffix}_{date_str}.md"
            
            filepath = notifier.save_report_to_file(
                f"{review_text['root_title']}\n\n{review_report}",
                report_filename
            )
            logger.info(f"大盘复盘报告已保存: {filepath}")
            
            # 推送通知（合并模式下跳过，由 main 层统一发送）
            if merge_notification and send_notification:
                logger.info("合并推送模式：跳过大盘复盘单独推送，将在个股+大盘复盘后统一发送")
            elif send_notification and notifier.is_available():
                # === 修改点：推送标题也加上市场后缀 ===
                push_title = f"{review_text['push_title']} ({market_suffix})"
                report_content = f"{push_title}\n\n{review_report}"

                # 显式传递 title 给 notifier.send
                success = notifier.send(report_content, email_send_to_all=True, title=push_title)
                if success:
                    logger.info(f"{market_suffix} 大盘复盘推送成功")
                else:
                    logger.warning(f"{market_suffix} 大盘复盘推送失败")
            elif not send_notification:
                logger.info("已跳过推送通知 (--no-notify)")
            
            return review_report
        
    except Exception as e:
        logger.error(f"大盘复盘分析失败: {e}")
    
    return None
