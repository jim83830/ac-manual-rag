"""讀取 manuals/manuals.yaml：每本說明書的檔名、代號（doc_id）與顯示名稱。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag.config import ConfigError

_DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_FIELDS = ("file", "doc_id", "name")


@dataclass(frozen=True)
class Manual:
    file: str
    doc_id: str
    name: str


def load_manuals(manuals_dir: Path) -> list[Manual]:
    registry = manuals_dir / "manuals.yaml"
    if not registry.exists():
        raise ConfigError(f"找不到說明書登記表 {registry}")
    entries = yaml.safe_load(registry.read_text(encoding="utf-8")) or []
    if not isinstance(entries, list):
        raise ConfigError("manuals.yaml 的最外層必須是清單（每本說明書一個以 - 開頭的項目）")

    manuals: list[Manual] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ConfigError(f"manuals.yaml 第 {i} 項格式錯誤")
        missing = [key for key in _FIELDS if not str(entry.get(key) or "").strip()]
        if missing:
            raise ConfigError(f"manuals.yaml 第 {i} 項缺少欄位：{', '.join(missing)}")
        manual = Manual(**{key: str(entry[key]).strip() for key in _FIELDS})
        if not _DOC_ID_RE.match(manual.doc_id):
            raise ConfigError(f"doc_id {manual.doc_id!r} 只能用小寫英文、數字和 -")
        if manual.doc_id in seen:
            raise ConfigError(f"doc_id {manual.doc_id!r} 重複")
        if not (manuals_dir / manual.file).exists():
            raise ConfigError(f"找不到說明書檔案 {manuals_dir / manual.file}")
        seen.add(manual.doc_id)
        manuals.append(manual)
    return manuals
