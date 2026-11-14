"""Disjoint seed namespaces and deterministic per-game agent seed derivation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

SeedNamespace = Literal["train", "dev", "eval"]
SEED_NAMESPACES: tuple[SeedNamespace, ...] = ("train", "dev", "eval")
_VALID_NAMESPACE_SET: frozenset[str] = frozenset(SEED_NAMESPACES)


def validate_namespace(namespace: str) -> SeedNamespace:
    """Validate and narrow a seed namespace string."""
    if namespace not in _VALID_NAMESPACE_SET:
        raise ValueError(
            f"Invalid seed namespace {namespace!r}; expected one of {SEED_NAMESPACES}"
        )
    return namespace  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SeedKey:
    """Seed key tagged with one of the disjoint namespaces (`train`, `dev`, `eval`)."""

    namespace: SeedNamespace
    identifier: str

    def __post_init__(self) -> None:
        validate_namespace(self.namespace)
        if not isinstance(self.identifier, str) or not self.identifier or self.identifier.strip() != self.identifier:
            raise ValueError(
                f"Invalid seed identifier {self.identifier!r}: must be a non-empty string without outer whitespace"
            )

    @property
    def key(self) -> str:
        """Canonical string passed to the adapter (`<namespace>-<identifier>`)."""
        return f"{self.namespace}-{self.identifier}"

    def __str__(self) -> str:
        return self.key

    @classmethod
    def parse(cls, raw: str) -> SeedKey:
        """Parse a canonical `<namespace>-<identifier>` seed key string."""
        if not isinstance(raw, str) or "-" not in raw:
            raise ValueError(
                f"Invalid seed key {raw!r}; expected '<namespace>-<identifier>' with namespace in {SEED_NAMESPACES}"
            )
        ns, ident = raw.split("-", 1)
        return cls(namespace=validate_namespace(ns), identifier=ident)


def make_seed_key(namespace: SeedNamespace, identifier: int | str) -> SeedKey:
    """Construct a validated `SeedKey` in the given namespace."""
    validate_namespace(namespace)
    if isinstance(identifier, int) and not isinstance(identifier, bool):
        if identifier < 0:
            raise ValueError(f"Seed index must be >= 0, got {identifier}")
        ident_str = f"{identifier:04d}"
    elif isinstance(identifier, str):
        ident_str = identifier
    else:
        raise TypeError(f"Seed identifier must be int or str, got {type(identifier).__name__}")
    return SeedKey(namespace=namespace, identifier=ident_str)


def generate_seed_keys(
    namespace: SeedNamespace,
    count: int,
    *,
    start: int = 0,
) -> tuple[SeedKey, ...]:
    """Generate `count` consecutive `SeedKey`s in `namespace` starting at `start`."""
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    validate_namespace(namespace)
    return tuple(make_seed_key(namespace, start + i) for i in range(count))


def derive_agent_seed(
    seed_key: SeedKey | str,
    *,
    map_name: str = "",
    role: str = "a",
) -> int:
    """Deterministically derive an unsigned 31-bit per-game agent RNG seed from a seed key."""
    canonical = seed_key.key if isinstance(seed_key, SeedKey) else SeedKey.parse(seed_key).key
    payload = f"{canonical}|{map_name}|{role}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
