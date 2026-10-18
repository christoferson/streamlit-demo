"""
cmn/tools/tool/tool_safety.py
=============================
Limits for tools whose input comes from the model, shared by the v3.7.5 tools in this
package and the legacy cmn/bedrock_converse_tools*.py tools.

- evaluate_expression(): arithmetic with size limits, so "9**9**9" can't hang the app
- check_url_is_public() / fetch_public_url(): only http(s) URLs on public addresses, checked
  on every redirect hop, so a tool can't reach localhost, the private network or the cloud
  metadata endpoint (http://169.254.169.254, which can hand out AWS credentials)

No streamlit.
"""

import ast
import ipaddress
import math
import operator as op
import socket
from urllib.parse import urljoin, urlparse

import requests


# ── Arithmetic ────────────────────────────────────────────────────────────────

_OPERATORS = {
    ast.Add:    op.add,
    ast.Sub:    op.sub,
    ast.Mult:   op.mul,
    ast.Div:    op.truediv,
    ast.Pow:    op.pow,
    ast.BitXor: op.xor,
    ast.USub:   op.neg,
}

_FUNCTIONS = {"round": round}

MAX_EXPRESSION_LENGTH = 1000
# Integer arithmetic is exact and unbounded in Python. Results must also stay under Python's
# 4,300-digit int-to-str limit, or turning the result into JSON for the model raises ValueError.
MAX_RESULT_DIGITS = 4_000


def evaluate_expression(expression: str):
    """
    Evaluate an arithmetic expression (+ - * / ** ^, unary minus, round()).
    Raises on invalid input, unsupported syntax, or a result over MAX_RESULT_DIGITS digits.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression is longer than {MAX_EXPRESSION_LENGTH} characters")
    return _evaluate(ast.parse(expression, mode="eval").body)


def _evaluate(node):
    if isinstance(node, ast.Constant):
        # Numbers only: a string constant would allow e.g. 'a' * 10**9
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise TypeError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            _check_power(left, right)
        return _check_result(_operator(node.op)(left, right))
    if isinstance(node, ast.UnaryOp):
        return _check_result(_operator(node.op)(_evaluate(node.operand)))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise NameError(f"Unsupported function: {ast.unparse(node.func)}")
        args = [_evaluate(arg) for arg in node.args]
        keywords = {kw.arg: _evaluate(kw.value) for kw in node.keywords}
        return _check_result(_FUNCTIONS[node.func.id](*args, **keywords))
    raise TypeError(f"Unsupported syntax: {type(node).__name__}")


def _operator(ast_operator):
    if type(ast_operator) not in _OPERATORS:
        raise TypeError(f"Unsupported operator: {type(ast_operator).__name__}")
    return _OPERATORS[type(ast_operator)]


def _check_power(base, exponent):
    """Refuse int ** int before computing it if the result would be too large."""
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0 and abs(base) > 1:
        if exponent * math.log10(abs(base)) > MAX_RESULT_DIGITS:
            raise ValueError(f"Result would have more than {MAX_RESULT_DIGITS:,} digits")


def _check_result(value):
    """Stop integers from growing past the limit through repeated operations."""
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_DIGITS * math.log2(10):
        raise ValueError(f"Result has more than {MAX_RESULT_DIGITS:,} digits")
    return value


# ── URLs ──────────────────────────────────────────────────────────────────────

MAX_REDIRECTS = 5
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024

# Content types read as text; anything else (zip, images, video, ...) is refused before
# downloading the body
TEXT_CONTENT_TYPES = ("application/xhtml+xml", "application/xml", "application/json")


class BlockedUrlError(Exception):
    """The URL is not loaded: not http/https, not a public address, too many redirects, or not text."""


def check_url_is_public(url: str) -> None:
    """
    Raise BlockedUrlError unless url is http(s) and its host resolves only to public addresses.
    Known gap: requests resolves the host again when connecting, so a DNS record that changes
    in between (DNS rebinding) is not caught.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedUrlError(f"Only http and https URLs can be loaded, not '{parsed.scheme or url}'")
    if not parsed.hostname:
        raise BlockedUrlError("The URL has no host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)}
    except socket.gaierror:
        raise BlockedUrlError(f"Could not resolve host {parsed.hostname}")
    for address_text in addresses:
        address = ipaddress.ip_address(address_text.split("%")[0])  # drop an IPv6 zone index
        if address.version == 6 and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global or address.is_multicast:
            raise BlockedUrlError(f"{parsed.hostname} resolves to a non-public address ({address})")


def fetch_public_url(url: str, headers: dict, timeout: float = 10):
    """
    GET url with check_url_is_public() on every redirect hop, refusing non-text responses and
    reading at most MAX_DOWNLOAD_BYTES. Returns (response, body bytes); the response is closed.
    """
    for _ in range(MAX_REDIRECTS + 1):
        check_url_is_public(url)
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=False, stream=True)
        if not response.is_redirect:
            break
        response.close()
        url = urljoin(url, response.headers["Location"])
    else:
        raise BlockedUrlError(f"More than {MAX_REDIRECTS} redirects")

    with response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type and not (content_type.startswith("text/") or content_type in TEXT_CONTENT_TYPES):
            raise BlockedUrlError(f"the response is {content_type}, not a web page or text")
        body = b""
        for chunk in response.iter_content(chunk_size=64 * 1024):
            body += chunk
            if len(body) >= MAX_DOWNLOAD_BYTES:
                body = body[:MAX_DOWNLOAD_BYTES]  # callers cut the text far shorter anyway
                break
    return response, body


def response_charset(response):
    """The charset the server declared, or None so the HTML parser detects it (e.g. from <meta>)."""
    return response.encoding if "charset" in response.headers.get("Content-Type", "").lower() else None
