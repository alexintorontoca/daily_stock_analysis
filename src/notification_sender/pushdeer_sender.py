import requests
import logging

logger = logging.getLogger(__name__)

class PushDeerSender:
    def __init__(self, config):
        """
        适配系统框架：从统一的 config 对象中读取配置
        """
        self.pushkey = getattr(config, 'pushdeer_key', None)
        self.url = "https://api2.getpushdeer.com/message/push"

    def _is_pushdeer_configured(self) -> bool:
        """提供给 NotificationService 用于自动探测渠道"""
        return bool(self.pushkey)

    def send_to_pushdeer(self, content: str, title: str = "A股自选股日报") -> bool:
        """
        适配系统调用名：send_to_pushdeer
        """
        if not self.pushkey:
            logger.error("PushDeer pushkey 未配置。")
            return False
        
        data = {
            "pushkey": self.pushkey,
            "text": title,
            "desp": content,
            "type": "markdown"
        }
        try:
            # 建议增加 timeout 防止网络问题导致主程序卡死
            response = requests.post(self.url, data=data, timeout=10)
            result = response.json()
            if result.get("code") == 0:
                logger.info("PushDeer 通知发送成功。")
                return True
            else:
                logger.error(f"PushDeer 发送失败: {result}")
                return False
        except Exception as e:
            logger.error(f"发送 PushDeer 通知时发生错误: {e}")
            return False
