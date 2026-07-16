from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from xtranslate.backend.app import create_app
    from xtranslate.backend.config import Settings
else:
    from xtranslate.backend.app import create_app
    from xtranslate.backend.config import Settings


def main() -> None:
    """启动服务。"""
    settings = Settings.from_env()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
