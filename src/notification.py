# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 通知层
===================================

职责：
1. 汇总分析结果生成日报
2. 支持 Markdown 格式输出
3. 多渠道推送（自动识别）：
    - 企业微信 Webhook
    - 飞书 Webhook
    - Telegram Bot
    - 邮件 SMTP
    - Pushover（手机/桌面推送）
    - PushDeer（新增）
"""
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from src.config import get_config
from src.analyzer import AnalysisResult
from src.enums import ReportType
from src.report_language import (
    get_localized_stock_name,
    get_report_labels,
    get_signal_level,
    localize_chip_health,
    localize_operation_advice,
    localize_trend_prediction,
    normalize_report_language,
)
from bot.models import BotMessage
from src.utils.data_processing import normalize_model_used
from src.notification_sender import (
    AstrbotSender,
    CustomWebhookSender,
    DiscordSender,
    EmailSender,
    FeishuSender,
    PushoverSender,
    PushplusSender,
    Serverchan3Sender,
    SlackSender,
    TelegramSender,
    WechatSender,
    # === 导入你刚才写的类 ===
    PushDeerSender, 
    WECHAT_IMAGE_MAX_BYTES
)

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """通知渠道类型"""
    WECHAT = "wechat"      # 企业微信
    FEISHU = "feishu"      # 飞书
    TELEGRAM = "telegram"  # Telegram
    EMAIL = "email"        # 邮件
    PUSHOVER = "pushover"  # Pushover（手机/桌面推送）
    PUSHPLUS = "pushplus"  # PushPlus（国内推送服务）
    SERVERCHAN3 = "serverchan3"  # Server酱3（手机APP推送服务）
    PUSHDEER = "pushdeer"  # PushDeer (iOS/macOS推送)
    CUSTOM = "custom"      # 自定义 Webhook
    DISCORD = "discord"    # Discord 机器人 (Bot)
    SLACK = "slack"        # Slack
    ASTRBOT = "astrbot"
    UNKNOWN = "unknown"    # 未知


class ChannelDetector:
    """
    渠道检测器 - 简化版
    
    根据配置直接判断渠道类型（不再需要 URL 解析）
    """
    
    @staticmethod
    def get_channel_name(channel: NotificationChannel) -> str:
        """获取渠道中文名称"""
        names = {
            NotificationChannel.WECHAT: "企业微信",
            NotificationChannel.FEISHU: "飞书",
            NotificationChannel.TELEGRAM: "Telegram",
            NotificationChannel.EMAIL: "邮件",
            NotificationChannel.PUSHOVER: "Pushover",
            NotificationChannel.PUSHPLUS: "PushPlus",
            NotificationChannel.SERVERCHAN3: "Server酱3",
            NotificationChannel.PUSHDEER: "PushDeer",
            NotificationChannel.CUSTOM: "自定义Webhook",
            NotificationChannel.DISCORD: "Discord机器人",
            NotificationChannel.SLACK: "Slack",
            NotificationChannel.ASTRBOT: "ASTRBOT机器人",
            NotificationChannel.UNKNOWN: "未知渠道",
        }
        return names.get(channel, "未知渠道")


class NotificationService(
    AstrbotSender,
    CustomWebhookSender,
    DiscordSender,
    EmailSender,
    FeishuSender,
    PushoverSender,
    PushplusSender,
    Serverchan3Sender,
    SlackSender,
    TelegramSender,
    WechatSender,
    # === 继承新发送器 ===
    PushDeerSender
):
    """
    通知服务
    """
    
    def __init__(self, source_message: Optional[BotMessage] = None):
        """
        初始化通知服务
        """
        config = get_config()
        self._source_message = source_message
        self._context_channels: List[str] = []

        # Markdown 转图片相关配置
        self._markdown_to_image_channels = set(
            getattr(config, 'markdown_to_image_channels', []) or []
        )
        self._markdown_to_image_max_chars = getattr(
            config, 'markdown_to_image_max_chars', 15000
        )

        self._report_summary_only = getattr(config, 'report_summary_only', False)
        self._history_compare_cache: Dict[Tuple[int, Tuple[Tuple[str, str], ...]], Dict[str, List[Dict[str, Any]]]] = {}

        # 初始化各基类发送器
        AstrbotSender.__init__(self, config)
        CustomWebhookSender.__init__(self, config)
        DiscordSender.__init__(self, config)
        EmailSender.__init__(self, config)
        FeishuSender.__init__(self, config)
        PushoverSender.__init__(self, config)
        PushplusSender.__init__(self, config)
        Serverchan3Sender.__init__(self, config)
        SlackSender.__init__(self, config)
        TelegramSender.__init__(self, config)
        WechatSender.__init__(self, config)
        # === 初始化 PushDeer ===
        # 注意：这里假设你的 config 类中有 pushdeer_key 字段
        PushDeerSender.__init__(self, getattr(config, 'pushdeer_key', None))

        # 检测所有已配置的渠道
        self._available_channels = self._detect_all_channels()
        if self._has_context_channel():
            self._context_channels.append("会话回复")

        if not self._available_channels and not self._context_channels:
            logger.warning("未配置有效的通知渠道，将不发送推送通知")
        else:
            channel_names = [ChannelDetector.get_channel_name(ch) for ch in self._available_channels]
            channel_names.extend(self._context_channels)
            logger.info(f"已配置 {len(channel_names)} 个通知渠道：{', '.join(channel_names)}")

    def _detect_all_channels(self) -> List[NotificationChannel]:
        """检测所有已配置的渠道"""
        channels = []
        
        # 1. 已有的渠道检测逻辑...
        if self._wechat_url:
            channels.append(NotificationChannel.WECHAT)
        if self._feishu_url:
            channels.append(NotificationChannel.FEISHU)
        if self._is_telegram_configured():
            channels.append(NotificationChannel.TELEGRAM)
        if self._is_email_configured():
            channels.append(NotificationChannel.EMAIL)
        if self._is_pushover_configured():
            channels.append(NotificationChannel.PUSHOVER)
        if self._pushplus_token:
            channels.append(NotificationChannel.PUSHPLUS)
        if self._serverchan3_sendkey:
            channels.append(NotificationChannel.SERVERCHAN3)
        
        # === 新增：检测 PushDeer ===
        if hasattr(self, 'pushkey') and self.pushkey:
            channels.append(NotificationChannel.PUSHDEER)

        # 其他 Webhook/Discord 等检测...
        if self._custom_webhook_urls:
            channels.append(NotificationChannel.CUSTOM)
        if self._is_discord_configured():
            channels.append(NotificationChannel.DISCORD)
        if self._is_slack_configured():
            channels.append(NotificationChannel.SLACK)
        if self._is_astrbot_configured():
            channels.append(NotificationChannel.ASTRBOT)
        if hasattr(self, 'pushkey') and self.pushkey:
            channels.append(NotificationChannel.PUSHDEER)
            
        return channels

    # ... 原有的 generate_daily_report, send_to_context 等方法保持不变 ...

    def push_report(self, title: str, content: str) -> bool:
        """
        这个方法负责遍历所有可用渠道并发送。
        """
        success_count = 0
        for channel in self._available_channels:
            try:
                res = False
                # === 新增：PushDeer 发送分支 ===
                if channel == NotificationChannel.PUSHDEER:
                    # 注意：调用的是你 pushdeer_sender.py 里的发送方法
                    res = self.send_to_pushdeer(content) 
                
                # --- 以下是原始仓库中其他渠道可能的逻辑 (请根据你文件实际情况补全) ---
                elif channel == NotificationChannel.WECHAT:
                    res = self.send_to_wechat(content)
                elif channel == NotificationChannel.FEISHU:
                    res = self.send_to_feishu(content)
                # ... 其他 channel 照旧 ...

                if res:
                    success_count += 1
            except Exception as e:
                logger.error(f"发送到 {channel.value} 失败: {e}")

        return success_count > 0
