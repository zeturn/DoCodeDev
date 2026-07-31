from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .relay_provider import make_relay_response, write_atomic_json


class RelayResponder:
    """Stateless external responder for the filesystem relay.

    It watches ``requests_dir`` for request packets, produces a model decision
    through a backend (``codex`` or ``mock``), and writes a response packet to
    ``responses_dir``. The responder is intentionally *dumb*: it never reads the
    workspace, the hidden checker, or any gold answer, and it never touches the
    model's credentials. The Codex backend simply shells out to the ``codex``
    executable, which manages its own auth via its own configuration.
    """

    def __init__(
        self,
        requests_dir: str,
        responses_dir: str,
        *,
        backend: str = "codex",
        model: str | None = None,
        codex_executable: str = "codex",
        codex_extra_args: list[str] | None = None,
        codex_timeout: float = 1800.0,
        poll_interval: float = 0.5,
        isolation_level: str = "relay_stateless_isolated",
        once: bool = False,
    ) -> None:
        self.requests_dir = Path(requests_dir)
        self.responses_dir = Path(responses_dir)
        self.backend = backend
        self.model = model
        self.codex_executable = codex_executable
        self.codex_extra_args = list(codex_extra_args or [])
        self.codex_timeout = float(codex_timeout)
        self.poll_interval = float(poll_interval)
        self.isolation_level = isolation_level
        self.once = once
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def process_one(self) -> bool:
        """Process a single pending request. Returns True if one was handled."""
        for path in sorted(self.requests_dir.glob("*.json")):
            if path.name.endswith(".processing.json") or path.name.endswith(".tmp"):
                continue
            processing = path.with_name(path.name[: -len(".json")] + ".processing.json")
            try:
                path.rename(processing)
            except FileNotFoundError:
                continue
            try:
                request = json.loads(processing.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                processing.unlink(missing_ok=True)
                continue
            try:
                response = self._answer(request)
                write_atomic_json(self.responses_dir / f"{request['request_id']}.json", response)
            finally:
                processing.unlink(missing_ok=True)
            return True
        return False

    def watch(self, max_iterations: int | None = None) -> None:
        it = 0
        while True:
            if max_iterations is not None and it >= max_iterations:
                return
            it += 1
            if self.process_one():
                if self.once:
                    return
                continue
            if self.once:
                return
            time.sleep(self.poll_interval)

    def _answer(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            if self.backend == "mock":
                content = self._mock_content(request)
            elif self.backend == "codex":
                content = self._codex_content(request)
            else:
                return make_relay_response(
                    request,
                    content=f"unknown backend: {self.backend}",
                    finish_reason="error",
                    responder=self.backend,
                    isolation_level=self.isolation_level,
                )
            return make_relay_response(
                request,
                content=content,
                finish_reason="stop",
                responder=self.backend,
                isolation_level=self.isolation_level,
            )
        except Exception as exc:  # noqa: BLE001
            return make_relay_response(
                request,
                content=f"{type(exc).__name__}: {exc}",
                finish_reason="error",
                responder=self.backend,
                isolation_level=self.isolation_level,
            )

    def _codex_content(self, request: dict[str, Any]) -> str:
        # The relay request carries a fully rendered DoCode decision prompt that
        # already instructs the model to respond as JSON. We forward it verbatim
        # to Codex. We do NOT pass any workspace paths, checker output, or gold
        # answers, and we never read Codex's credentials/token files.
        prompt = request.get("prompt") or request.get("system") or ""
        cmd = [self.codex_executable, "exec", prompt]
        if self.model:
            cmd += ["-m", self.model]
        cmd += self.codex_extra_args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.codex_timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"codex_failed rc={result.returncode} stderr={result.stderr[:500]}"
            )
        return result.stdout

    def _mock_content(self, request: dict[str, Any]) -> str:
        # A trivial valid final_candidate used for transport smoke tests.
        return json.dumps(
            {
                "type": "final_candidate",
                "summary": "mock relay responder final candidate",
                "verification": "",
                "no_test_reason": None,
                "remaining_risks": [],
            },
            ensure_ascii=False,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DoCode external relay responder (stateless model endpoint).")
    parser.add_argument("--requests", required=True, help="Directory containing relay request packets.")
    parser.add_argument("--responses", required=True, help="Directory to write relay response packets.")
    parser.add_argument("--backend", choices=["codex", "mock"], default="codex")
    parser.add_argument("--model", default=None, help="Model id passed to the backend (e.g. a Codex model).")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--codex-arg", action="append", default=[], help="Extra args forwarded to codex.")
    parser.add_argument("--codex-timeout", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--isolation-level", default="relay_stateless_isolated")
    parser.add_argument("--once", action="store_true", help="Process a single pending request then exit.")
    parser.add_argument("--watch", action="store_true", help="Watch for requests until interrupted.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    responder = RelayResponder(
        args.requests,
        args.responses,
        backend=args.backend,
        model=args.model,
        codex_executable=args.codex_executable,
        codex_extra_args=args.codex_arg,
        codex_timeout=args.codex_timeout,
        poll_interval=args.poll_interval,
        isolation_level=args.isolation_level,
        once=args.once,
    )
    if args.once:
        responder.watch(max_iterations=1)
        return 0
    responder.watch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
