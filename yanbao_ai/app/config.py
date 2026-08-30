"""配置加载：读 config.toml + 环境变量，[llm] 端点可回落读 ~/.claude/settings.json。

设计要点：
- 密钥只来自 env 或未跟踪的 config.toml，绝不写入库、绝不日志打印其值。
- [llm].base_url / key 留空时，自动读 Claude Code 官方配置 ~/.claude/settings.json
  的 env.ANTHROPIC_BASE_URL / env.ANTHROPIC_AUTH_TOKEN——cc-switch 切哪家就跟到哪家。
"""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# 项目根：yanbao_ai/（本文件在 yanbao_ai/yanbao/config.py）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 仓库根：包含 mineru_pipeline/ 与 yanbao_ai/ 的上一层。
# 一切默认路径都相对文件位置派生，整棵树搬到别处也不会失效。
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"

# mineru 产物根（catalog 从这里消费）。均可被 config.toml 的 [paths] 覆盖。
DEFAULT_CANONICAL = REPO_ROOT / "mineru_pipeline" / "canonical"
DEFAULT_MANIFEST = REPO_ROOT / "mineru_pipeline" / "manifest" / "manifest.jsonl"
DEFAULT_DB = PROJECT_ROOT / "data" / "yanbao.db"


def _redact(value: str | None) -> str:
    """把 key 脱敏成可安全打印的形式（只留前后各 4 位）。"""
    if not value:
        return "<未设置>"
    if len(value) <= 12:
        return "****"
    return f"{value[:4]}…{value[-4:]}"


@dataclass
class LLMConfig:
    """Claude（生成/抽取/重排），走中转站。"""

    base_url: str | None = None
    api_key: str | None = None
    model_gen: str = "claude-opus"
    model_cheap: str = "claude-haiku"
    source: str = "config.toml"  # 记录端点来源：config.toml 或 settings.json

    @property
    def key_redacted(self) -> str:
        return _redact(self.api_key)


@dataclass
class EmbedConfig:
    """向量化：OpenAI 兼容端点（qwen3.7-text-embedding @ 1024 维 或 OpenAI 官方）。"""

    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    model: str = "text-embedding-3-large"
    dimensions: int = 1024
    batch_size: int = 20  # qwen 兼容端点硬上限 20；OpenAI 可放大
    # 总闸门：false → 全系统不再调嵌入端点，检索走纯 BM25，index 不写向量。
    # **与"key 是否存在"解耦**：key 可以原样留在 config.toml（备查/待恢复），
    # 只要 enabled=false 就一次请求都不发。2026-08-27 qwen key 被封（401
    # API-key is blocked）后置为 false —— 之前只靠"删 key"来关，会逼人动凭证。
    enabled: bool = True

    @property
    def usable(self) -> bool:
        """能否真正调嵌入端点：开关打开 **且** 有 key。所有调用方统一问这一个。"""
        return self.enabled and bool(self.api_key)

    @property
    def key_redacted(self) -> str:
        return _redact(self.api_key)


@dataclass
class Paths:
    canonical: Path = DEFAULT_CANONICAL
    manifest: Path = DEFAULT_MANIFEST
    db: Path = DEFAULT_DB

    def require_inputs(self) -> None:
        """catalog 的输入前置校验：canonical/ 与 manifest 必须存在，否则大声报错。

        路径来自默认（相对仓库根派生）或 config.toml [paths] 覆盖；无论哪种，
        缺失都直接抛错并指明该配哪个键，绝不静默继续跑出一个空库。
        """
        missing = []
        if not self.canonical.is_dir():
            missing.append(
                f"  canonical 目录不存在: {self.canonical}\n"
                f"    → 在 config.toml 的 [paths].canonical 指定正确路径"
            )
        if not self.manifest.is_file():
            missing.append(
                f"  manifest 文件不存在: {self.manifest}\n"
                f"    → 在 config.toml 的 [paths].manifest 指定正确路径"
            )
        if missing:
            raise FileNotFoundError(
                "catalog 输入缺失，无法继续：\n" + "\n".join(missing)
            )


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    paths: Paths = field(default_factory=Paths)


def _read_claude_settings() -> tuple[str | None, str | None]:
    """从 ~/.claude/settings.json 读 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN。

    读不到（文件缺失/无该字段）时返回 (None, None)，不报错。
    """
    try:
        data = json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    env = data.get("env", {}) if isinstance(data, dict) else {}
    return env.get("ANTHROPIC_BASE_URL"), env.get("ANTHROPIC_AUTH_TOKEN")


def load_config(path: Path | None = None) -> Config:
    """加载配置。config.toml 不存在时用全默认值（阶段 0 catalog 不需要任何 key）。"""
    cfg_path = path or DEFAULT_CONFIG
    raw: dict = {}
    if cfg_path.exists():
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    cfg = Config()

    # ---- [paths] ----
    p = raw.get("paths", {})
    if p.get("canonical"):
        cfg.paths.canonical = Path(p["canonical"])
    if p.get("manifest"):
        cfg.paths.manifest = Path(p["manifest"])
    if p.get("db"):
        cfg.paths.db = Path(p["db"])

    # ---- [llm] ----
    llm = raw.get("llm", {})
    cfg.llm.base_url = llm.get("base_url") or None
    # key 优先取 env（api_key_env 指定的变量），其次直接读 config.toml 的 api_key。
    # 直填 api_key 便于本机使用（env 变量在 PowerShell 会话间不持久）；config.toml 已 gitignore。
    cfg.llm.api_key = (
        os.environ.get(llm.get("api_key_env", "")) or llm.get("api_key") or None
    )
    cfg.llm.model_gen = llm.get("model_gen", cfg.llm.model_gen)
    cfg.llm.model_cheap = llm.get("model_cheap", cfg.llm.model_cheap)

    # [llm] 端点跟随：base_url 或 key 缺失 → 回落读 Claude Code settings.json
    if not cfg.llm.base_url or not cfg.llm.api_key:
        s_url, s_key = _read_claude_settings()
        if not cfg.llm.base_url and s_url:
            cfg.llm.base_url = s_url
            cfg.llm.source = "settings.json"
        if not cfg.llm.api_key and s_key:
            cfg.llm.api_key = s_key
            cfg.llm.source = "settings.json"

    # ---- [embed] ----
    emb = raw.get("embed", {})
    cfg.embed.base_url = emb.get("base_url", cfg.embed.base_url)
    cfg.embed.api_key = (
        os.environ.get(emb.get("api_key_env", "")) or emb.get("api_key") or None
    )
    cfg.embed.model = emb.get("model", cfg.embed.model)
    cfg.embed.dimensions = int(emb.get("dimensions", cfg.embed.dimensions))
    cfg.embed.batch_size = int(emb.get("batch_size", cfg.embed.batch_size))
    cfg.embed.enabled = bool(emb.get("enabled", cfg.embed.enabled))

    return cfg


class InputsMissingError(RuntimeError):
    """catalog 所需的 mineru 产物不存在时抛出，附带排查提示。"""


def require_inputs(cfg: Config) -> None:
    """catalog 运行前的硬校验：canonical 目录与 manifest 缺任一即报错退出。

    不做静默兜底——路径错了就要立刻、明确地失败，而不是跑出一个空库。
    """
    problems: list[str] = []
    if not cfg.paths.canonical.is_dir():
        problems.append(f"  canonical 目录不存在：{cfg.paths.canonical}")
    if not cfg.paths.manifest.is_file():
        problems.append(f"  manifest 文件不存在：{cfg.paths.manifest}")
    if problems:
        raise InputsMissingError(
            "找不到 mineru 产物，无法建 catalog：\n"
            + "\n".join(problems)
            + "\n请确认 mineru_pipeline 已产出，或在 config.toml 的 [paths] 下"
            "显式指定 canonical / manifest 路径。"
        )
