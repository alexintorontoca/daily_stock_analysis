# -*- coding: utf-8 -*-
"""
Email 发送提醒服务

职责：
1. 通过 SMTP 发送 Email 消息（发件人+密码齐全时优先走 SMTP）
2. 无 SMTP 配置且设置 RESEND_PROXY_URL 时，改走 Cloudflare resend-proxy Worker 的 GET 接口
   （参数 to / subject / text，发件人由 Worker 固定，不需要 SMTP 密码）
"""
import logging
from typing import Optional, List
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header
from email.utils import formataddr
import os
import smtplib
import urllib.parse
import urllib.request

from data_provider.base import normalize_stock_code
from src.config import Config
from src.formatters import markdown_to_html_document, strip_hidden_markdown_metadata


logger = logging.getLogger(__name__)


# SMTP 服务器配置（自动识别）
SMTP_CONFIGS = {
    # QQ邮箱
    "qq.com": {"server": "smtp.qq.com", "port": 465, "ssl": True},
    "foxmail.com": {"server": "smtp.qq.com", "port": 465, "ssl": True},
    # 网易邮箱
    "163.com": {"server": "smtp.163.com", "port": 465, "ssl": True},
    "126.com": {"server": "smtp.126.com", "port": 465, "ssl": True},
    # Gmail
    "gmail.com": {"server": "smtp.gmail.com", "port": 587, "ssl": False},
    # Outlook
    "outlook.com": {"server": "smtp-mail.outlook.com", "port": 587, "ssl": False},
    "hotmail.com": {"server": "smtp-mail.outlook.com", "port": 587, "ssl": False},
    "live.com": {"server": "smtp-mail.outlook.com", "port": 587, "ssl": False},
    # 新浪
    "sina.com": {"server": "smtp.sina.com", "port": 465, "ssl": True},
    # 搜狐
    "sohu.com": {"server": "smtp.sohu.com", "port": 465, "ssl": True},
    # 阿里云
    "aliyun.com": {"server": "smtp.aliyun.com", "port": 465, "ssl": True},
    # 139邮箱
    "139.com": {"server": "smtp.139.com", "port": 465, "ssl": True},
    # Resend：autni.eu.org 发件域名走 Resend SMTP，登录名固定为 resend，密码为 Resend API Key
    "autni.eu.org": {"server": "smtp.resend.com", "port": 465, "ssl": True, "username": "resend"},
}


class EmailSender:
    
    def __init__(self, config: Config):
        """
        初始化 Email 配置

        Args:
            config: 配置对象
        """
        self._email_config = {
            'sender': config.email_sender,
            'sender_name': getattr(config, 'email_sender_name', 'daily_stock_analysis股票分析助手'),
            'password': config.email_password,
            'receivers': config.email_receivers or ([config.email_sender] if config.email_sender else []),
            'resend_proxy_url': (os.environ.get('RESEND_PROXY_URL') or getattr(config, 'resend_proxy_url', None) or '').strip() or None,
        }
        self._stock_email_groups = getattr(config, 'stock_email_groups', None) or []
        
    def _is_email_configured(self) -> bool:
        """检查邮件配置是否完整：SMTP 需发件人+密码；无 SMTP 时 resend-proxy 只需地址+收件人"""
        if self._email_config['sender'] and self._email_config['password']:
            return True
        if self._email_config.get('resend_proxy_url'):
            return bool(self._email_config['receivers'])
        return False
    
    def get_receivers_for_stocks(self, stock_codes: List[str]) -> List[str]:
        """
        Look up email receivers for given stock codes based on stock_email_groups.
        Returns union of receivers for all matching groups; falls back to default if none match.
        Stock codes are canonicalized before comparison so that equivalent
        formats (e.g. SH600519 vs 600519) match correctly.
        """
        if not stock_codes or not self._stock_email_groups:
            return self._email_config['receivers']
        normalized_codes = [normalize_stock_code(c) for c in stock_codes]
        seen: set = set()
        result: List[str] = []
        for stocks, emails in self._stock_email_groups:
            for code in normalized_codes:
                if code in stocks:
                    for e in emails:
                        if e not in seen:
                            seen.add(e)
                            result.append(e)
                    break
        return result if result else self._email_config['receivers']

    def get_all_email_receivers(self) -> List[str]:
        """
        Return union of all configured email receivers (all groups + default).
        Used for market review which should go to everyone.
        """
        seen: set = set()
        result: List[str] = []
        for _, emails in self._stock_email_groups:
            for e in emails:
                if e not in seen:
                    seen.add(e)
                    result.append(e)
        for e in self._email_config['receivers']:
            if e not in seen:
                seen.add(e)
                result.append(e)
        return result

    def _format_sender_address(self, sender: str) -> str:
        """Encode display name safely so non-ASCII sender names work across SMTP providers."""
        sender_name = self._email_config.get('sender_name') or '股票分析助手'
        return formataddr((str(Header(str(sender_name), 'utf-8')), sender))

    @staticmethod
    def _close_server(server: Optional[smtplib.SMTP]) -> None:
        """Best-effort SMTP cleanup to avoid leaving sockets open on header/build errors.

        Exceptions from quit()/close() are intentionally silenced — connection may already
        be in a broken state, and there is nothing useful to do at this point.
        """
        if server is None:
            return
        try:
            server.quit()
        except Exception:
            try:
                server.close()
            except Exception:
                pass
    
    def _send_via_resend_proxy(
        self,
        content: str,
        subject: str,
        receivers: List[str],
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """经 resend-proxy Worker 的 GET 接口发信（Worker 固定发件人，纯文本）。

        Worker 只读查询参数 to / subject / text，一次请求只接受一个收件人，
        故逐个收件人调用；任一收件人失败即返回 False。
        """
        proxy_url = (self._email_config.get('resend_proxy_url') or '').rstrip('/')
        for receiver in receivers:
            params = urllib.parse.urlencode(
                {'to': receiver, 'subject': subject, 'text': content}
            )
            url = f"{proxy_url}?{params}"
            try:
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=timeout_seconds or 30) as resp:
                    resp.read()
                    if resp.status != 200:
                        logger.error(f"resend-proxy 发送失败：收件人 {receiver}，状态 {resp.status}")
                        return False
            except Exception as e:
                logger.error(f"resend-proxy 发送失败：收件人 {receiver}，{e}")
                return False
        logger.info(f"resend-proxy 邮件发送成功，收件人: {receivers}")
        return True

    def send_to_email(
        self,
        content: str,
        subject: Optional[str] = None,
        receivers: Optional[List[str]] = None,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """
        通过 SMTP 发送邮件（自动识别 SMTP 服务器）
        
        Args:
            content: 邮件内容（支持 Markdown，会转换为 HTML）
            subject: 邮件主题（可选，默认自动生成）
            receivers: 收件人列表（可选，默认使用配置的 receivers）
            
        Returns:
            是否发送成功
        """
        if not self._is_email_configured():
            logger.warning("邮件配置不完整，跳过推送")
            return False

        receivers = receivers or self._email_config['receivers']
        sender = self._email_config['sender']
        password = self._email_config['password']
        # SMTP 优先：发件人+密码齐全时走 SMTP，不再走代理，避免代理的 URL 长度限制
        if self._email_config.get('resend_proxy_url') and not (sender and password):
            if subject is None:
                date_str = datetime.now().strftime('%Y-%m-%d')
                subject = f"📈 股票智能分析报告 - {date_str}"
            sanitized_content = strip_hidden_markdown_metadata(content).strip()
            return self._send_via_resend_proxy(
                sanitized_content, subject, receivers, timeout_seconds=timeout_seconds
            )
        
        receivers = receivers or self._email_config['receivers']
        server: Optional[smtplib.SMTP] = None
        
        try:
            # 生成主题
            if subject is None:
                date_str = datetime.now().strftime('%Y-%m-%d')
                subject = f"📈 股票智能分析报告 - {date_str}"

            sanitized_content = strip_hidden_markdown_metadata(content).strip()
            
            # 将 Markdown 转换为简单 HTML
            html_content = markdown_to_html_document(sanitized_content)
            
            # 构建邮件
            msg = MIMEMultipart('alternative')
            msg['Subject'] = Header(subject, 'utf-8')
            msg['From'] = self._format_sender_address(sender)
            msg['To'] = ', '.join(receivers)
            
            # 添加纯文本和 HTML 两个版本
            text_part = MIMEText(sanitized_content, 'plain', 'utf-8')
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(text_part)
            msg.attach(html_part)
            
            # 自动识别 SMTP 配置
            domain = sender.split('@')[-1].lower()
            smtp_config = SMTP_CONFIGS.get(domain)
            
            if smtp_config:
                smtp_server = smtp_config['server']
                smtp_port = smtp_config['port']
                use_ssl = smtp_config['ssl']
                logger.info(f"自动识别邮箱类型: {domain} -> {smtp_server}:{smtp_port}")
            else:
                # 未知邮箱，尝试通用配置
                smtp_server = f"smtp.{domain}"
                smtp_port = 465
                use_ssl = True
                logger.warning(f"未知邮箱类型 {domain}，尝试通用配置: {smtp_server}:{smtp_port}")
            
            # 根据配置选择连接方式
            if use_ssl:
                # SSL 连接（端口 465）
                server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=timeout_seconds or 30)
            else:
                # TLS 连接（端口 587）
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=timeout_seconds or 30)
                server.starttls()
            
            # Resend SMTP 的登录名固定为 resend（不是发件地址），其余邮箱沿用发件地址
            login_username = (smtp_config or {}).get('username', sender)
            server.login(login_username, password)
            server.send_message(msg)
            
            logger.info(f"邮件发送成功，收件人: {receivers}")
            return True
            
        except smtplib.SMTPAuthenticationError:
            logger.error("邮件发送失败：认证错误，请检查邮箱和授权码是否正确")
            return False
        except smtplib.SMTPConnectError as e:
            logger.error(f"邮件发送失败：无法连接 SMTP 服务器 - {e}")
            return False
        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
            return False
        finally:
            self._close_server(server)

    def _send_email_with_inline_image(
        self, image_bytes: bytes, receivers: Optional[List[str]] = None
    ) -> bool:
        """Send email with inline image attachment (Issue #289)."""
        if not self._is_email_configured():
            return False
        if self._email_config.get('resend_proxy_url') and not (self._email_config['sender'] and self._email_config['password']):
            logger.warning("resend-proxy 只支持纯文本，图片版邮件未发送（请勿在 Actions 里开启转图渠道）")
            return False
        sender = self._email_config['sender']
        password = self._email_config['password']
        receivers = receivers or self._email_config['receivers']
        server: Optional[smtplib.SMTP] = None
        try:
            date_str = datetime.now().strftime('%Y-%m-%d')
            subject = f"📈 股票智能分析报告 - {date_str}"
            msg = MIMEMultipart('related')
            msg['Subject'] = Header(subject, 'utf-8')
            msg['From'] = self._format_sender_address(sender)
            msg['To'] = ', '.join(receivers)

            alt = MIMEMultipart('alternative')
            alt.attach(MIMEText('报告已生成，详见下方图片。', 'plain', 'utf-8'))
            html_body = (
                '<p>报告已生成，详见下方图片（点击可查看大图）：</p>'
                '<p><img src="cid:report-image" alt="股票分析报告" style="max-width:100%%;" /></p>'
            )
            alt.attach(MIMEText(html_body, 'html', 'utf-8'))
            msg.attach(alt)

            img_part = MIMEImage(image_bytes, _subtype='png')
            img_part.add_header('Content-Disposition', 'inline', filename='report.png')
            img_part.add_header('Content-ID', '<report-image>')
            msg.attach(img_part)

            domain = sender.split('@')[-1].lower()
            smtp_config = SMTP_CONFIGS.get(domain)
            if smtp_config:
                smtp_server, smtp_port = smtp_config['server'], smtp_config['port']
                use_ssl = smtp_config['ssl']
            else:
                smtp_server, smtp_port = f"smtp.{domain}", 465
                use_ssl = True

            if use_ssl:
                server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
            else:
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
                server.starttls()
            server.login((smtp_config or {}).get('username', sender), password)
            server.send_message(msg)
            logger.info("邮件（内联图片）发送成功，收件人: %s", receivers)
            return True
        except Exception as e:
            logger.error("邮件（内联图片）发送失败: %s", e)
            return False
        finally:
            self._close_server(server)
