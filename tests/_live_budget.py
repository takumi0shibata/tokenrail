"""Fail-closed transport for opt-in live tests; never part of the public API."""

from __future__ import annotations

import json
import threading
from collections import Counter

import httpx

MODEL = "gpt-6-luna"
TOKEN_BUDGET = 100_000


class LiveSafetyStop(RuntimeError):
    pass


def _tokens(value: object) -> bool:
    return type(value) is int and value >= 0


class LiveBudgetTransport(httpx.BaseTransport):
    """Serialize preflight/reservation/send/accounting, including batch workers.

    Reservations are never refunded, even after a timeout or missing usage.
    Check each response's payer before allowing another request. The total
    token ceiling bounds this run; it does not guarantee free usage.
    """

    def __init__(self, inner: httpx.BaseTransport):
        self.inner = inner
        self._lock = threading.Lock()
        self.report = {
            "model": MODEL, "requests": 0, "preflight_requests": 0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cached_tokens": 0, "cache_write_tokens": 0,
            "reserved_tokens": 0,
            "payer": None, "stop_reason": None, "statuses": Counter(),
        }

    def _stop(self, reason: str) -> None:
        self.report["stop_reason"] = reason
        raise LiveSafetyStop(reason)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            if self.report["stop_reason"]:
                raise LiveSafetyStop(self.report["stop_reason"])
            if request.method != "POST" or str(request.url) != "https://api.openai.com/v1/responses":
                self._stop("Unexpected live API route")
            try:
                body = json.loads(request.content)
            except (ValueError, UnicodeError):
                self._stop("Invalid generation payload")
            allowed = {
                "model", "input", "instructions", "reasoning", "text", "max_output_tokens",
                "temperature", "top_p", "metadata", "service_tier", "store",
                "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention",
            }
            if not isinstance(body, dict) or body.keys() - allowed or body.get("model") != MODEL:
                self._stop("Unexpected live model or request fields")
            cap = body.get("max_output_tokens")
            if not _tokens(cap) or not 1 <= cap <= 256 or body.get("reasoning") != {"effort": "none"}:
                self._stop("Missing or unsafe generation limits")
            probe = self.report["requests"] == 0
            if probe:
                if body.get("input") != "OK" or cap != 16 or body.get("text") or body.get("instructions"):
                    self._stop("First generation must be the minimal payer probe")

            # Count the exact SDK-generated input, including structured schemas.
            count_body = {key: body[key] for key in ("model", "input", "instructions", "reasoning", "text")
                          if key in body}
            headers = {key: value for key, value in request.headers.items() if key.lower() != "content-length"}
            count_request = httpx.Request(
                "POST", "https://api.openai.com/v1/responses/input_tokens",
                headers=headers, json=count_body, extensions=dict(request.extensions),
            )
            self.report["preflight_requests"] += 1
            try:
                counted = self.inner.handle_request(count_request)
                try:
                    counted.read()
                    count_data = counted.json() if counted.status_code == 200 else None
                finally:
                    counted.close()
            except Exception:
                self._stop("Input-token preflight failed; no generation sent")
            if not isinstance(count_data, dict) or not _tokens(count_data.get("input_tokens")):
                self._stop("Input-token count missing or invalid; no generation sent")
            input_tokens = count_data["input_tokens"]
            if probe and input_tokens > 128:
                self._stop("Probe input unexpectedly large; no generation sent")
            reservation = input_tokens + cap
            if self.report["reserved_tokens"] + reservation > TOKEN_BUDGET:
                self._stop("Total token budget exhausted; no generation sent")
            self.report["reserved_tokens"] += reservation
            self.report["requests"] += 1
            try:
                response = self.inner.handle_request(request)
                response.read()
            except Exception:
                self._stop("Generation transport failed; reservation retained and further sends blocked")
            try:
                raw = response.json()
            except (ValueError, UnicodeError):
                raw = None
            self._account(raw, response.status_code, input_tokens, cap)
            return response

    def _account(self, raw: object, status_code: int, input_bound: int, output_bound: int) -> None:
        if not isinstance(raw, dict):
            self.report["stop_reason"] = "Generation body unreadable; further sends blocked"
            return
        billing = raw.get("billing")
        payer = billing.get("payer") if isinstance(billing, dict) else None
        self.report["payer"] = payer if payer in ("openai", "developer") else "unknown"
        if payer != "openai":
            self.report["stop_reason"] = "Payer is not openai or is unknown; further sends blocked"
        usage = raw.get("usage")
        keys = ("input_tokens", "output_tokens", "total_tokens")
        if not isinstance(usage, dict) or any(not _tokens(usage.get(key)) for key in keys):
            self.report["stop_reason"] = "Generation usage missing or invalid; reservation retained"
            return
        if (usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
                or usage["input_tokens"] > input_bound or usage["output_tokens"] > output_bound):
            self.report["stop_reason"] = "Generation usage exceeds or contradicts preflight bounds"
            return
        for key in keys:
            self.report[key] += usage[key]
        # Cache and reasoning counts are subsets, never added to the total.
        details = usage.get("input_tokens_details")
        if isinstance(details, dict):
            for key in ("cached_tokens", "cache_write_tokens"):
                if _tokens(details.get(key)):
                    self.report[key] += details[key]
        self.report["statuses"][str(raw.get("status") or status_code)] += 1
        if status_code != 200 or raw.get("model") != MODEL:
            self.report["stop_reason"] = "Unexpected generation status or model; further sends blocked"

    def close(self) -> None:
        self.inner.close()
