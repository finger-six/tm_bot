"""Engine-derived card registry and canonical card-name resolution."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

DEFAULT_CARD_REGISTRY_PATH = Path("data") / "card_registry.json"

# Verified spelling differences between BGA/TFMStats logs and the pinned engine CardName enum.
# Aliases only resolve when their canonical target exists in the loaded registry.
_BGA_ALIASES: dict[str, str] = {
    "biolabs": "Biolab",
    "ceo's favourite project": "CEO's Favorite Project",
    "excentric sponsor": "Eccentric Sponsor",
}


def _norm_key(name: str) -> str:
    cleaned = unicodedata.normalize("NFKC", name).strip()
    cleaned = cleaned.replace("\u2019", "'")
    return " ".join(cleaned.split()).casefold()


class CardRegistry:
    """Resolves raw log card names against the pinned engine's card registry."""

    def __init__(
        self,
        names: Iterable[str],
        *,
        card_modules: dict[str, str] | None = None,
        card_kinds: dict[str, str] | None = None,
    ) -> None:
        self._exact: set[str] = set()
        self._by_key: dict[str, str] = {}
        for raw in names:
            canonical = str(raw).strip()
            if not canonical:
                continue
            self._exact.add(canonical)
            self._by_key.setdefault(_norm_key(canonical), canonical)

        self._aliases: dict[str, str] = {}
        for alias_key, target in _BGA_ALIASES.items():
            resolved_target = self._by_key.get(_norm_key(target))
            if resolved_target is not None:
                self._aliases[alias_key] = resolved_target

        self.card_modules: dict[str, str] = dict(card_modules or {})
        self.card_kinds: dict[str, str] = dict(card_kinds or {})

    @classmethod
    def from_names(cls, names: Iterable[str]) -> CardRegistry:
        return cls(names)

    @classmethod
    def from_json(cls, path: Path | str = DEFAULT_CARD_REGISTRY_PATH) -> CardRegistry:
        reg_path = Path(path)
        if not reg_path.is_file():
            raise FileNotFoundError(
                f"Card registry JSON does not exist: {reg_path}. "
                "Generate it with: node tools/export_card_registry.cjs"
            )
        payload = json.loads(reg_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return cls(payload)
        if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
            return cls(
                payload["cards"],
                card_modules=payload.get("card_modules") if isinstance(payload.get("card_modules"), dict) else None,
                card_kinds=payload.get("card_kinds") if isinstance(payload.get("card_kinds"), dict) else None,
            )
        raise ValueError(f"Unsupported card registry JSON format in {reg_path}")

    def __len__(self) -> int:
        return len(self._exact)

    def resolve(self, name: str | None) -> str | None:
        """Resolve a raw card name to its canonical engine CardName, or None if unknown."""
        if name is None or not isinstance(name, str):
            return None
        stripped = name.strip()
        if not stripped:
            return None
        if stripped in self._exact:
            return stripped
        key = _norm_key(stripped)
        if key in self._by_key:
            return self._by_key[key]
        return self._aliases.get(key)
