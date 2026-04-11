# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 LLM 调用逻辑（通过 LiteLLM 统一调用 Gemini/Anthropic/OpenAI 等）
2. 结合技术面和消息面生成分析报告
3. 解析 LLM 响应为结构化 AnalysisResult
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.agent.skills.defaults import CORE_TRADING_SKILL_POLICY_ZH
from src.config import (
    Config,
    extra_litellm_params,
    get_api_keys_for_model,
    get_config,
    get_configured_llm_models,
    resolve_news_window_days,
)
from src.storage import persist_llm_usage
from src.data.stock_mapping import STOCK_NAME_MAP
from src.report_language import (
    get_signal_level,
    get_no_data_text,
    get_placeholder_text,
    get_unknown_text,
    infer_decision_type_from_advice,
    localize_chip_health,
    localize_confidence_level,
    normalize_report_language,
)
from src.schemas.report_schema import AnalysisReportSchema
from src.market_context import get_market_role, get_market_guidelines

logger = logging.getLogger(__name__)


class _LiteLLMStreamError(RuntimeError):
    """Internal error wrapper that records whether any text was streamed."""

    def __init__(self, message: str, *, partial_received: bool = False):
        super().__init__(message)
        self.partial_received = partial_received


def check_content_integrity(result: "AnalysisResult") -> Tuple[bool, List[str]]:
    """
    Check mandatory fields for report content integrity.
    Returns (pass, missing_fields). Module-level for use by pipeline (agent weak mode).
    """
    missing: List[str] = []
    if result.sentiment_score is None:
        missing.append("sentiment_score")
    advice = result.operation_advice
    if not advice or not isinstance(advice, str) or not advice.strip():
        missing.append("operation_advice")
    summary = result.analysis_summary
    if not summary or not isinstance(summary, str) or not summary.strip():
        missing.append("analysis_summary")
    dash = result.dashboard if isinstance(result.dashboard, dict) else {}
    core = dash.get("core_conclusion")
    core = core if isinstance(core, dict) else {}
    if not (core.get("one_sentence") or "").strip():
        missing.append("dashboard.core_conclusion.one_sentence")
    intel = dash.get("intelligence")
    intel = intel if isinstance(intel, dict) else None
    if intel is None or "risk_alerts" not in intel:
        missing.append("dashboard.intelligence.risk_alerts")
    if result.decision_type in ("buy", "hold"):
        battle = dash.get("battle_plan")
        battle = battle if isinstance(battle, dict) else {}
        sp = battle.get("sniper_points")
        sp = sp if isinstance(sp, dict) else {}
        stop_loss = sp.get("stop_loss")
        if stop_loss is None or (isinstance(stop_loss, str) and not stop_loss.strip()):
            missing.append("dashboard.battle_plan.sniper_points.stop_loss")
    return len(missing) == 0, missing


def apply_placeholder_fill(result: "AnalysisResult", missing_fields: List[str]) -> None:
    """Fill missing mandatory fields with placeholders (in-place). Module-level for pipeline."""
    placeholder = get_placeholder_text(getattr(result, "report_language", "zh"))
    for field in missing_fields:
        if field == "sentiment_score":
            result.sentiment_score = 50
        elif field == "operation_advice":
            result.operation_advice = result.operation_advice or placeholder
        elif field == "analysis_summary":
            result.analysis_summary = result.analysis_summary or placeholder
        elif field == "dashboard.core_conclusion.one_sentence":
            if not result.dashboard:
                result.dashboard = {}
            if "core_conclusion" not in result.dashboard:
                result.dashboard["core_conclusion"] = {}
            result.dashboard["core_conclusion"]["one_sentence"] = (
                result.dashboard["core_conclusion"].get("one_sentence") or placeholder
            )
        elif field == "dashboard.intelligence.risk_alerts":
            if not result.dashboard:
                result.dashboard = {}
            if "intelligence" not in result.dashboard:
                result.dashboard["intelligence"] = {}
            if "risk_alerts" not in result.dashboard["intelligence"]:
                result.dashboard["intelligence"]["risk_alerts"] = []
        elif field == "dashboard.battle_plan.sniper_points.stop_loss":
            if not result.dashboard:
                result.dashboard = {}
            if "battle_plan" not in result.dashboard:
                result.dashboard["battle_plan"] = {}
            if "sniper_points" not in result.dashboard["battle_plan"]:
                result.dashboard["battle_plan"]["sniper_points"] = {}
            result.dashboard["battle_plan"]["sniper_points"]["stop_loss"] = placeholder


# ---------- chip_structure fallback (Issue #589) ----------

_CHIP_KEYS: tuple = ("profit_ratio", "avg_cost", "concentration", "chip_health")


def _is_value_placeholder(v: Any) -> bool:
    """True if value is empty or placeholder (N/A, 数据缺失, etc.)."""
    if v is None:
        return True
    if isinstance(v, (int, float)) and v == 0:
        return True
    s = str(v).strip().lower()
    return s in ("", "n/a", "na", "数据缺失", "未知", "data unavailable", "unknown", "tbd")


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Safely convert to float; return default on failure. Private helper for chip fill."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        try:
            return default if math.isnan(float(v)) else float(v)
        except (ValueError, TypeError):
            return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _derive_chip_health(profit_ratio: float, concentration_90: float, language: str = "zh") -> str:
    """Derive chip_health from profit_ratio and concentration_90."""
    if profit_ratio >= 0.9:
        return localize_chip_health("警惕", language)  # 获利盘极高
    if concentration_90 >= 0.25:
        return localize_chip_health("警惕", language)  # 筹码分散
    if concentration_90 < 0.15 and 0.3 <= profit_ratio < 0.9:
        return localize_chip_health("健康", language)  # 集中且获利比例适中
    return localize_chip_health("一般", language)


def _build_chip_structure_from_data(chip_data: Any, language: str = "zh") -> Dict[str, Any]:
    """Build chip_structure dict from ChipDistribution or dict."""
    if hasattr(chip_data, "profit_ratio"):
        pr = _safe_float(chip_data.profit_ratio)
        ac = chip_data.avg_cost
        c90 = _safe_float(chip_data.concentration_90)
    else:
        d = chip_data if isinstance(chip_data, dict) else {}
        pr = _safe_float(d.get("profit_ratio"))
        ac = d.get("avg_cost")
        c90 = _safe_float(d.get("concentration_90"))
    chip_health = _derive_chip_health(pr, c90, language=language)
    return {
        "profit_ratio": f"{pr:.1%}",
        "avg_cost": ac if (ac is not None and _safe_float(ac) != 0.0) else "N/A",
        "concentration": f"{c90:.2%}",
        "chip_health": chip_health,
    }


def fill_chip_structure_if_needed(result: "AnalysisResult", chip_data: Any) -> None:
    """When chip_data exists, fill chip_structure placeholder fields from chip_data (in-place)."""
    if not result or not chip_data:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        cs = dp.get("chip_structure") or {}
        filled = _build_chip_structure_from_data(
            chip_data,
            language=getattr(result, "report_language", "zh"),
        )
        merged = dict(cs)
        for k in _CHIP_KEYS:
            if _is_value_placeholder(merged.get(k)):
                merged[k] = filled[k]
        if merged != cs:
            dp["chip_structure"] = merged
            logger.info("[chip_structure] Filled placeholder chip fields from data source (Issue #589)")
    except Exception as e:
        logger.warning("[chip_structure] Fill failed, skipping: %s", e)


_PRICE_POS_KEYS = ("ma5", "ma10", "ma20", "bias_ma5", "bias_status", "current_price", "support_level", "resistance_level")


def fill_price_position_if_needed(
    result: "AnalysisResult",
    trend_result: Any = None,
    realtime_quote: Any = None,
) -> None:
    """Fill missing price_position fields from trend_result / realtime data (in-place)."""
    if not result:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        pp = dp.get("price_position") or {}

        computed: Dict[str, Any] = {}
        if trend_result:
            tr = trend_result if isinstance(trend_result, dict) else (
                trend_result.__dict__ if hasattr(trend_result, "__dict__") else {}
            )
            computed["ma5"] = tr.get("ma5")
            computed["ma10"] = tr.get("ma10")
            computed["ma20"] = tr.get("ma20")
            computed["bias_ma5"] = tr.get("bias_ma5")
            computed["current_price"] = tr.get("current_price")
            support_levels = tr.get("support_levels") or []
            resistance_levels = tr.get("resistance_levels") or []
            if support_levels:
                computed["support_level"] = support_levels[0]
            if resistance_levels:
                computed["resistance_level"] = resistance_levels[0]
        if realtime_quote:
            rq = realtime_quote if isinstance(realtime_quote, dict) else (
                realtime_quote.to_dict() if hasattr(realtime_quote, "to_dict") else {}
            )
            if _is_value_placeholder(computed.get("current_price")):
                computed["current_price"] = rq.get("price")

        filled = False
        for k in _PRICE_POS_KEYS:
            if _is_value_placeholder(pp.get(k)) and not _is_value_placeholder(computed.get(k)):
                pp[k] = computed[k]
                filled = True
        if filled:
            dp["price_position"] = pp
            logger.info("[price_position] Filled placeholder fields from computed data")
    except Exception as e:
        logger.warning("[price_position] Fill failed, skipping: %s", e)


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """多来源获取股票中文名称"""
    if context:
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']

    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]

    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"无法初始化 DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"从数据源获取股票名称失败: {e}")

    return f'股票{stock_code}'


@dataclass
class AnalysisResult:
    """AI 分析结果数据类 - 决策仪表盘版"""
    code: str
    name: str
    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    decision_type: str = "hold"
    confidence_level: str = "中"
    report_language: str = "zh"
    dashboard: Optional[Dict[str, Any]] = None
    trend_analysis: str = ""
    short_term_outlook: str = ""
    medium_term_outlook: str = ""
    technical_analysis: str = ""
    ma_analysis: str = ""
    volume_analysis: str = ""
    pattern_analysis: str = ""
    fundamental_analysis: str = ""
    sector_position: str = ""
    company_highlights: str = ""
    news_summary: str = ""
    market_sentiment: str = ""
    hot_topics: str = ""
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    buy_reason: str = ""
    market_snapshot: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    search_performed: bool = False
    data_sources: str = ""
    success: bool = True
    error_message: Optional[str] = None
    current_price: Optional[float] = None
    change_pct: Optional[float] = None
    model_used: Optional[str] = None
    query_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'report_language': self.report_language,
            'dashboard': self.dashboard,
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
            'model_used': self.model_used,
        }

    def get_core_conclusion(self) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        _, emoji, _ = get_signal_level(
            self.operation_advice,
            self.sentiment_score,
            self.report_language,
        )
        return emoji

    def get_confidence_stars(self) -> str:
        star_map = {
            "高": "⭐⭐⭐", "high": "⭐⭐⭐",
            "中": "⭐⭐", "medium": "⭐⭐",
            "低": "⭐", "low": "⭐",
        }
        return star_map.get(str(self.confidence_level or "").strip().lower(), "⭐⭐")


class GeminiAnalyzer:
    """Gemini AI 分析器 (LiteLLM 驱动)"""

    LEGACY_DEFAULT_SYSTEM_PROMPT = """你是一位专注于趋势交易的{market_placeholder}投资分析师，负责生成专业的【决策仪表盘】分析报告。

{guidelines_placeholder}

""" + CORE_TRADING_SKILL_POLICY_ZH + """
... (此处省略你源码中已有的 LEGACY JSON 定义以节省篇幅，实际应用时请保留)
"""

    SYSTEM_PROMPT = """你是一位{market_placeholder}投资分析师，负责生成专业的【决策仪表盘】分析报告。

{guidelines_placeholder}

{default_skill_policy_section}
{skills_section}

## 输出格式：决策仪表盘 JSON
... (此处省略你源码中已有的完整 JSON 结构定义以节省篇幅，实际应用时请保留)
"""

    TEXT_SYSTEM_PROMPT = """你是一位专业的股票分析助手。
- 回答必须基于用户提供的数据与上下文
- 若信息不足，要明确指出不确定性
- 不要编造价格、财报或新闻事实
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        config: Optional[Config] = None,
        skills: Optional[List[str]] = None,
        skill_instructions: Optional[str] = None,
        default_skill_policy: Optional[str] = None,
        use_legacy_default_prompt: Optional[bool] = None,
    ):
        self._config_override = config
        self._requested_skills = list(skills) if skills is not None else None
        self._skill_instructions_override = skill_instructions
        self._default_skill_policy_override = default_skill_policy
        self._use_legacy_default_prompt_override = use_legacy_default_prompt
        self._resolved_prompt_state: Optional[Dict[str, Any]] = None
        self._router = None
        self._litellm_available = False
        self._init_litellm()
        if not self._litellm_available:
            logger.warning("No LLM configured (LITELLM_MODEL / API keys), AI analysis will be unavailable")

    def _init_litellm(self):
        """初始化 LiteLLM 路由"""
        cfg = self._get_runtime_config()
        model_list = get_configured_llm_models(cfg)
        if not model_list:
            return
        try:
            self._router = Router(
                model_list=model_list,
                routing_strategy="latency-based-routing",
                set_verbose=False
            )
            self._litellm_available = True
        except Exception as e:
            logger.error(f"Failed to initialize LiteLLM Router: {e}")

    def _get_runtime_config(self) -> Config:
        return getattr(self, "_config_override", None) or get_config()

    def _get_skill_prompt_sections(self) -> tuple[str, str, bool]:
        """解析技能指令和策略"""
        skill_instructions = getattr(self, "_skill_instructions_override", None)
        default_skill_policy = getattr(self, "_default_skill_policy_override", None)
        use_legacy_default_prompt = getattr(self, "_use_legacy_default_prompt_override", None)

        if skill_instructions is not None and default_skill_policy is not None:
            return (skill_instructions, default_skill_policy, bool(use_legacy_default_prompt))

        resolved_state = getattr(self, "_resolved_prompt_state", None)
        if resolved_state is None:
            from src.agent.factory import resolve_skill_prompt_state
            prompt_state = resolve_skill_prompt_state(
                self._get_runtime_config(),
                skills=getattr(self, "_requested_skills", None),
            )
            resolved_state = {
                "skill_instructions": prompt_state.skill_instructions,
                "default_skill_policy": prompt_state.default_skill_policy,
                "use_legacy_default_prompt": bool(getattr(prompt_state, "use_legacy_default_prompt", False)),
            }
            self._resolved_prompt_state = resolved_state

        return (
            skill_instructions if skill_instructions is not None else resolved_state.get("skill_instructions", ""),
            default_skill_policy if default_skill_policy is not None else resolved_state.get("default_skill_policy", ""),
            use_legacy_default_prompt if use_legacy_default_prompt is not None else resolved_state.get("use_legacy_default_prompt", False)
        )

    def _get_analysis_system_prompt(self, report_language: str, stock_code: str = "") -> str:
        """构建分析系统提示词"""
        lang = normalize_report_language(report_language)
        market_role = get_market_role(stock_code, lang)
        market_guidelines = get_market_guidelines(stock_code, lang)
        skill_instructions, default_skill_policy, use_legacy_default_prompt = self._get_skill_prompt_sections()
        
        if use_legacy_default_prompt:
            base_prompt = self.LEGACY_DEFAULT_SYSTEM_PROMPT.replace("{market_placeholder}", market_role).replace("{guidelines_placeholder}", market_guidelines)
        else:
            skills_section = f"## 激活的交易技能\n\n{skill_instructions}\n" if skill_instructions else ""
            default_skill_policy_section = f"{default_skill_policy}\n" if default_skill_policy else ""
            base_prompt = (
                self.SYSTEM_PROMPT.replace("{market_placeholder}", market_role)
                .replace("{guidelines_placeholder}", market_guidelines)
                .replace("{default_skill_policy_section}", default_skill_policy_section)
                .replace("{skills_section}", skills_section)
            )
        
        if lang == "en":
            return base_prompt + "\n\n## Output Language (highest priority)\n- Keep all JSON keys unchanged.\n- All human-readable values in English."
        return base_prompt

    def analyze(self, context: Dict[str, Any], report_language: str = "zh") -> AnalysisResult:
        """执行 AI 分析"""
        stock_code = context.get('stock_code', 'Unknown')
        stock_name = get_stock_name_multi_source(stock_code, context)
        
        if not self._litellm_available:
            return AnalysisResult(code=stock_code, name=stock_name, sentiment_score=50, 
                                 trend_prediction="未知", operation_advice="无法分析", 
                                 success=False, error_message="LLM未配置")

        system_prompt = self._get_analysis_system_prompt(report_language, stock_code)
        user_prompt = f"请分析股票：{stock_name} ({stock_code})\n\n上下文数据：\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        try:
            cfg = self._get_runtime_config()
            model = cfg.litellm_model
            
            # 调用 LiteLLM
            response = self._router.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                **extra_litellm_params(model, cfg)
            )

            raw_content = response.choices[0].message.content
            # 使用 json_repair 处理可能的截断或格式错误
            data = json.loads(repair_json(raw_content))

            result = AnalysisResult(
                code=stock_code,
                name=stock_name,
                sentiment_score=data.get('sentiment_score', 50),
                trend_prediction=data.get('trend_prediction', '震荡'),
                operation_advice=data.get('operation_advice', '持币观望'),
                decision_type=data.get('decision_type', 'hold'),
                dashboard=data.get('dashboard'),
                analysis_summary=data.get('analysis_summary', ''),
                report_language=report_language,
                model_used=model,
                raw_response=raw_content
            )

            # 自动填充逻辑
            fill_chip_structure_if_needed(result, context.get('chip_distribution'))
            fill_price_position_if_needed(result, context.get('trend'), context.get('realtime'))
            
            # 完整性检查
            is_valid, missing = check_content_integrity(result)
            if not is_valid:
                apply_placeholder_fill(result, missing)

            return result

        except Exception as e:
            logger.error(f"Analysis failed for {stock_code}: {e}")
            return AnalysisResult(code=stock_code, name=stock_name, sentiment_score=50, 
                                 trend_prediction="错误", operation_advice="分析出错", 
                                 success=False, error_message=str(e))
