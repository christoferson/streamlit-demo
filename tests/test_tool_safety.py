"""
Tests for the limits in cmn/tools/tool/tool_safety.py and the tools that use them.
No network: DNS lookups and HTTP requests are faked.

Run from the project root:  python -m unittest discover -s tests -v
"""
import json
import socket
import unittest
from unittest.mock import patch

import requests

try:
    from cmn.tools.tool import tool_safety
    from cmn.tools.tool import CalculatorBedrockConverseTool as V3Calculator
    from cmn.tools.tool.bedrock_converse_tools_tool_aws_docs import is_aws_docs_url
    from cmn.bedrock_converse_tools import CalculatorBedrockConverseTool as LegacyCalculator
    IMPORT_ERROR = None
except Exception as err:  # e.g. Python 3.8: cmn/tools/tool/__init__.py imports modules with 3.10+ syntax
    IMPORT_ERROR = err

skip_if_unavailable = unittest.skipIf(IMPORT_ERROR is not None, f"cmn.tools.tool not importable: {IMPORT_ERROR}")


def addresses(*ips):
    """A fake socket.getaddrinfo result resolving to the given IPs."""
    return lambda host, port, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]


def http_response(status=200, headers=None, body=b""):
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    response._content = body
    response._content_consumed = True  # iter_content() then serves _content instead of reading raw
    response.raw = None
    return response


@skip_if_unavailable
class EvaluateExpressionTest(unittest.TestCase):

    def test_arithmetic(self):
        self.assertEqual(tool_safety.evaluate_expression("1234 * 5678"), 7006652)
        self.assertEqual(tool_safety.evaluate_expression("round(10 / 3, 2)"), 3.33)
        self.assertEqual(tool_safety.evaluate_expression("2 ** -2"), 0.25)
        self.assertEqual(tool_safety.evaluate_expression("-3 ** 2"), -9)

    def test_huge_power_is_refused_before_computing(self):
        # Without the limit this runs for hours
        with self.assertRaisesRegex(ValueError, "more than 4,000 digits"):
            tool_safety.evaluate_expression("9**9**9")

    def test_result_size_limit(self):
        self.assertEqual(len(str(tool_safety.evaluate_expression("10 ** 3999"))), 4000)
        with self.assertRaises(ValueError):
            tool_safety.evaluate_expression("10 ** 4000")
        with self.assertRaises(ValueError):
            tool_safety.evaluate_expression("(10**2000) * (10**2000)")

    def test_results_stay_json_serializable(self):
        # Python refuses int -> str past 4,300 digits; the limit keeps results under it
        json.dumps(tool_safety.evaluate_expression("10 ** 3999"))

    def test_only_numbers_and_arithmetic(self):
        for expression in ["'a' * 10**9", "True + 1", "__import__('os')", "round.__self__", "7 % 3"]:
            with self.subTest(expression=expression), self.assertRaises((TypeError, NameError)):
                tool_safety.evaluate_expression(expression)

    def test_expression_length_limit(self):
        with self.assertRaisesRegex(ValueError, "longer than"):
            tool_safety.evaluate_expression("1+" * 600 + "1")


@skip_if_unavailable
class CalculatorToolsTest(unittest.TestCase):

    def test_both_calculators_return_errors_instead_of_raising(self):
        for tool in (LegacyCalculator(), V3Calculator()):
            with self.subTest(tool=type(tool).__module__):
                result = tool.invoke(None, tool_args={"expression": "9**9**9"})
                self.assertIn("error", result)

    def test_result_formats_are_unchanged(self):
        self.assertEqual(LegacyCalculator().invoke(None, tool_args={"expression": "6 * 7"}), 42)
        self.assertEqual(V3Calculator().invoke(None, tool_args={"expression": "6 * 7"}), {"result": 42})


@skip_if_unavailable
class CheckUrlIsPublicTest(unittest.TestCase):

    def test_public_address_is_allowed(self):
        with patch("socket.getaddrinfo", addresses("93.184.216.34")):
            tool_safety.check_url_is_public("https://example.com/page")

    def test_internal_addresses_are_blocked(self):
        for ip in ["127.0.0.1", "169.254.169.254", "169.254.170.2", "10.0.0.1", "192.168.1.1",
                   "100.64.0.1", "0.0.0.0", "::1", "fe80::1", "::ffff:169.254.169.254", "224.0.0.1"]:
            with self.subTest(ip=ip), patch("socket.getaddrinfo", addresses(ip)):
                with self.assertRaises(tool_safety.BlockedUrlError):
                    tool_safety.check_url_is_public("http://some-host.example/")

    def test_any_internal_address_blocks(self):
        # A name resolving to a public and a private address is refused
        with patch("socket.getaddrinfo", addresses("93.184.216.34", "10.0.0.5")):
            with self.assertRaises(tool_safety.BlockedUrlError):
                tool_safety.check_url_is_public("http://mixed.example/")

    def test_only_http_and_https(self):
        for url in ["file:///etc/passwd", "ftp://example.com/", "gopher://example.com/", "example.com"]:
            with self.subTest(url=url), self.assertRaisesRegex(tool_safety.BlockedUrlError, "Only http and https"):
                tool_safety.check_url_is_public(url)


@skip_if_unavailable
class FetchPublicUrlTest(unittest.TestCase):

    def test_redirect_to_internal_address_is_blocked(self):
        responses = [http_response(302, {"Location": "http://169.254.169.254/latest/meta-data/"})]
        resolve = lambda host, port, **kwargs: addresses("169.254.169.254" if host == "169.254.169.254" else "93.184.216.34")(host, port)
        with patch("socket.getaddrinfo", resolve), patch("requests.get", side_effect=responses) as get:
            with self.assertRaisesRegex(tool_safety.BlockedUrlError, "169.254.169.254"):
                tool_safety.fetch_public_url("https://example.com/redirect", headers={})
        self.assertEqual(get.call_count, 1)  # the metadata endpoint was never requested
        self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_too_many_redirects(self):
        loop = [http_response(302, {"Location": "https://example.com/again"}) for _ in range(10)]
        with patch("socket.getaddrinfo", addresses("93.184.216.34")), patch("requests.get", side_effect=loop):
            with self.assertRaisesRegex(tool_safety.BlockedUrlError, "redirects"):
                tool_safety.fetch_public_url("https://example.com/", headers={})

    def test_non_text_response_is_refused(self):
        response = http_response(200, {"Content-Type": "application/octet-stream"}, b"\x00" * 100)
        with patch("socket.getaddrinfo", addresses("93.184.216.34")), patch("requests.get", return_value=response):
            with self.assertRaisesRegex(tool_safety.BlockedUrlError, "application/octet-stream"):
                tool_safety.fetch_public_url("https://example.com/file.bin", headers={})

    def test_download_is_capped(self):
        body = b"a" * (tool_safety.MAX_DOWNLOAD_BYTES + 1024 * 1024)
        response = http_response(200, {"Content-Type": "text/plain"}, body)
        with patch("socket.getaddrinfo", addresses("93.184.216.34")), patch("requests.get", return_value=response):
            _, downloaded = tool_safety.fetch_public_url("https://example.com/big.txt", headers={})
        self.assertEqual(len(downloaded), tool_safety.MAX_DOWNLOAD_BYTES)


@skip_if_unavailable
class AwsDocsUrlTest(unittest.TestCase):

    def test_only_docs_aws_amazon_com_over_https(self):
        self.assertTrue(is_aws_docs_url("https://docs.aws.amazon.com/lambda/latest/dg/welcome.html"))
        for url in [
            "https://docs.aws.amazon.com.evil.example/x",               # lookalike host
            "https://docs.aws.amazon.com@169.254.169.254/latest/",     # everything before '@' is a username
            "https://user:pw@docs.aws.amazon.com/x",
            "http://docs.aws.amazon.com/lambda/",
            "https://docs.aws.amazon.com:8443/x",
        ]:
            with self.subTest(url=url):
                self.assertFalse(is_aws_docs_url(url))


if __name__ == "__main__":
    unittest.main()
