"""Serve the coverage page locally: upload a clip, calibrate a frame, read the play.

  python scripts/serve.py            # http://127.0.0.1:8000
  python scripts/serve.py --port 8010
"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run("nfl_coverage.web.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
