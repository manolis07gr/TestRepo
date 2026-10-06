"""Check Kalshi API credentials end to end without ever printing them.

Reads KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY (or KALSHI_PRIVATE_KEY_PATH) from the
environment, loads the RSA key in whatever form it was pasted, opens the authenticated
market-data WebSocket and waits for the first message.

Usage: python scripts/check_kalshi_auth.py [--ws-url wss://...] [--timeout 15]
Exit codes: 0 authenticated, 1 credentials missing/unloadable, 2 rejected or unreachable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

from cma.adapters.kalshi import KALSHI_WS_URL, KalshiSigner, normalize_pem, ws_auth_headers
from cma.domain.time import SystemClock


def _describe_env() -> None:
    for name in ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY", "KALSHI_PRIVATE_KEY_PATH"):
        value = os.environ.get(name)
        print(f"{name}: {'not set' if value is None else f'set, {len(value)} characters'}")
    key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    uuid = re.fullmatch(r"[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", key_id)
    print(f"key id format: {'OK (UUID)' if uuid else 'UNEXPECTED'}")


def _load_signer() -> KalshiSigner | None:
    inline = os.environ.get("KALSHI_PRIVATE_KEY", "")
    if inline:
        pem = normalize_pem(inline)
        label = pem.splitlines()[0] if pem.startswith("-----BEGIN") else "unrecognised"
        print(f"private key form: {label}")
    try:
        from cryptography.hazmat.primitives import serialization

        if inline:
            key = serialization.load_pem_private_key(normalize_pem(inline).encode(), password=None)
            print(f"private key loads: {type(key).__name__}, {getattr(key, 'key_size', '?')} bits")
    except ImportError:
        print("cryptography is not installed: pip install 'cryptography>=42'")
        return None
    except (ValueError, TypeError) as exc:
        print(f"private key does NOT load: {type(exc).__name__}")
        return None
    return KalshiSigner.from_env(
        key_id_env="KALSHI_API_KEY_ID",
        private_key_env="KALSHI_PRIVATE_KEY",
        private_key_path_env="KALSHI_PRIVATE_KEY_PATH",
        clock=SystemClock(),
    )


async def _probe(signer: KalshiSigner, ws_url: str, timeout: float) -> int:
    import websockets

    headers = ws_auth_headers(signer, ws_url)
    try:
        async with websockets.connect(
            ws_url, additional_headers=headers, open_timeout=timeout
        ) as ws:
            print(f"websocket handshake: OK (authenticated) to {ws_url}")
            await ws.send(
                json.dumps({"id": 1, "cmd": "subscribe", "params": {"channels": ["ticker"]}})
            )
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                print(f"first message: type={msg.get('type')}")
            except TimeoutError:
                print("no message within the timeout (authentication still succeeded)")
            return 0
    except websockets.exceptions.InvalidStatus as exc:
        status = exc.response.status_code
        hint = " (key id / private key mismatch, or key revoked)" if status == 401 else ""
        print(f"websocket handshake REJECTED: HTTP {status}{hint}")
        return 2
    except (OSError, TimeoutError) as exc:
        print(f"websocket unreachable: {type(exc).__name__}: {exc}")
        return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-url", default=KALSHI_WS_URL)
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()
    _describe_env()
    signer = _load_signer()
    if signer is None:
        print("RESULT: credentials missing or unusable")
        return 1
    code = asyncio.run(_probe(signer, args.ws_url, args.timeout))
    print("RESULT: authenticated" if code == 0 else "RESULT: not authenticated")
    return code


if __name__ == "__main__":
    sys.exit(main())
