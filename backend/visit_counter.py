from __future__ import annotations

import json
from threading import Lock
from pathlib import Path


class VisitCounter:
    """持久化页面访问计数，避免服务重启后丢失。"""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._lock = Lock()
        self._count = 0
        self._load()

    def increment(self) -> int:
        with self._lock:
            self._count += 1
            self._persist()
            return self._count

    def get_count(self) -> int:
        with self._lock:
            return self._count

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return
        count = payload.get("total_visits", 0)
        if isinstance(count, int) and count >= 0:
            self._count = count

    def _persist(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as fp:
            json.dump({"total_visits": self._count}, fp, ensure_ascii=True)
        temp_path.replace(self.storage_path)
