"""Memory interface — shared base class for geometric and semantic memory modules."""
from __future__ import annotations


class MemoryInterface:
    """Shared API that both GeometricMemory and SemanticMemory implement.

    All methods raise NotImplementedError by default; subclasses override
    the ones they support. Queries return sensible defaults (empty/false)
    so callers don't crash when a subclass doesn't implement every method.
    """

    # ── core ──

    def update(
        self,
        metadata: dict,
        visible_objects: list[dict],
        action: str,
        success: bool,
        error_message: str | None,
        task_criteria: str = "",
        intent: str = "",
    ) -> None:
        raise NotImplementedError

    def render(self) -> str:
        raise NotImplementedError

    # ── object queries (used by dedup, phase detection, curiosity) ──

    def has_type(self, object_type: str) -> bool:
        return False

    def is_type_visible(self, object_type: str) -> bool:
        return False

    def get_type_distance(self, object_type: str) -> float | None:
        return None

    def get_all_object_types(self) -> set[str]:
        return set()

    # ── receptacle queries ──

    def get_visible_receptacles(self) -> list[str]:
        return []

    def get_remembered_receptacles(self) -> list[str]:
        return []

    def get_unvisited_receptacles(self, approached_types: set[str] | None = None) -> list[str]:
        return []

    def get_searched_receptacle_types(self) -> set[str]:
        return set()

    def get_type_distance_any(self, object_type: str) -> float | None:
        return None

    # ── searched markers ──

    def mark_searched(self, object_type: str | None = None, object_id: str | None = None) -> int:
        return 0

    def unmark_searched(self, object_type: str) -> int:
        return 0

    def unmark_searched_by_id(self, object_id: str) -> int:
        return 0

    # ── visitation tracking (curiosity scoreboard) ──

    def record_receptacle_visit(self, receptacle_type: str) -> None:
        pass

    def record_receptacle_open(self, receptacle_type: str) -> None:
        pass

    def record_object_discovered_in(self, receptacle_type: str) -> None:
        pass

    def get_receptacle_entries_for_curiosity(self) -> list[dict]:
        return []

    # ── phase detection helpers ──

    def get_remembered_type_distance(self, object_type: str) -> float | None:
        return None

    # ── stats (consumed by episode outcome) ──

    receptacle_visit_counts: dict[str, int]
    receptacle_open_counts: dict[str, int]
    objects_found_by_receptacle: dict[str, int]
