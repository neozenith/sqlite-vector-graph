"""Entry point: python -m server [--port 8200]."""

import argparse
import logging

import uvicorn

from server.config import DEFAULT_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


def main() -> None:
    """Run the uvicorn server."""
    parser = argparse.ArgumentParser(description="muninn-viz API server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    uvicorn.run("server.main:app", host=args.host, port=args.port, reload=True)


main()
