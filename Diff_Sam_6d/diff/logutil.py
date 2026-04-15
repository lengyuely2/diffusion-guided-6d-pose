"""训练时 tee 到文件，避免 conda run / nohup 长时间不刷新 stdout。"""

from __future__ import annotations

import sys
from pathlib import Path


def setup_tee_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logf = path.open("w", encoding="utf-8", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr

    class TeeIO:
        def __init__(self, stream, file):
            self._stream = stream
            self._file = file

        def write(self, data: str) -> int:
            self._stream.write(data)
            self._stream.flush()
            self._file.write(data)
            self._file.flush()
            return len(data)

        def flush(self) -> None:
            self._stream.flush()
            self._file.flush()

        def isatty(self) -> bool:
            return self._stream.isatty()

    sys.stdout = TeeIO(orig_out, logf)
    sys.stderr = TeeIO(orig_err, logf)
