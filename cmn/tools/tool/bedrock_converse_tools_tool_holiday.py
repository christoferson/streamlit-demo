import logging
import holidays
from datetime import date, datetime, timedelta
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)

# Supported countries — extendable
SUPPORTED_COUNTRIES = {
    "JP": "Japan",
    "US": "United States",
    "GB": "United Kingdom",
    "AU": "Australia",
    "CA": "Canada",
    "DE": "Germany",
    "FR": "France",
    "SG": "Singapore",
    "MY": "Malaysia",
    "TH": "Thailand",
}


class HolidayBedrockConverseTool(AbstractBedrockConverseTool):
    """
    Checks if a given date is a public holiday in a specific country.
    Uses the `holidays` library for accurate, up-to-date holiday data
    including moveable feasts and substitution holidays.
    """

    def __init__(self):
        name = "holiday_checker"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Checks if a specific date is a public holiday in a given country. "
                    "Supports moveable holidays and substitution days. "
                    "Use this when user asks about holidays for a specific country. "
                    "Call datetime tool first to get today's date, "
                    "then pass it here to check holidays."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": (
                                    "Date to check in YYYY-MM-DD format. "
                                    "Example: '2026-03-20'. "
                                    "Use today's date from datetime tool."
                                ),
                            },
                            "country_code": {
                                "type": "string",
                                "enum": list(SUPPORTED_COUNTRIES.keys()),
                                "description": (
                                    "ISO country code. "
                                    "Example: 'JP' for Japan, 'US' for United States."
                                ),
                            },
                            "days_ahead": {
                                "type": "integer",
                                "description": (
                                    "Optional. Check N days ahead from the given date. "
                                    "Example: 1 checks tomorrow, 7 checks next week. "
                                    "Defaults to 0 (just the given date)."
                                ),
                            },
                        },
                        "required": ["date", "country_code"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return (
            "holiday_checker : checks if a date is a public holiday in a specific country. "
            "Call datetime tool first to get today's date, then pass it here. "
            f"Supported countries: {', '.join(SUPPORTED_COUNTRIES.values())}."
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        args         = tool_args or {}
        date_str     = args.get("date")
        country_code = args.get("country_code", "US").upper()
        days_ahead   = int(args.get("days_ahead") or 0)

        logger.info(
            "HolidayTool invoked: date=%s country=%s days_ahead=%d",
            date_str, country_code, days_ahead,
        )

        if not date_str:
            return {"error": "date is required. Call datetime tool first."}

        if country_code not in SUPPORTED_COUNTRIES:
            return {
                "error":     f"Unsupported country: {country_code}",
                "supported": list(SUPPORTED_COUNTRIES.keys()),
            }

        try:
            check_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return {"error": f"Invalid date format: {date_str}. Use YYYY-MM-DD."}

        # ── Check date range ──────────────────────────────────────────────
        results = []
        for i in range(days_ahead + 1):
            target = check_date + timedelta(days=i)
            results.append(self._check_date(target, country_code))

        return {
            "country_code":    country_code,
            "country_name":    SUPPORTED_COUNTRIES[country_code],
            "checked_dates":   results,
            "summary": {
                "any_holiday": any(r["is_holiday"] or r["is_weekend"] for r in results),
                "holidays":    [r for r in results if r["is_holiday"]],
                "weekends":    [r for r in results if r["is_weekend"]],
                "working_days":[r for r in results if not r["is_holiday"] and not r["is_weekend"]],
            },
        }

    def _check_date(self, target: date, country_code: str) -> dict:
        """Check a single date for holiday and weekend status."""
        year         = target.year
        country_holidays = holidays.country_holidays(country_code, years=year)

        is_weekend      = target.weekday() >= 5
        holiday_name    = country_holidays.get(target)
        is_holiday      = holiday_name is not None

        return {
            "date":           target.strftime("%Y-%m-%d"),
            "day_of_week":    target.strftime("%A"),
            "is_weekend":     is_weekend,
            "is_holiday":     is_holiday,
            "holiday_name":   holiday_name,
            "is_day_off":     is_weekend or is_holiday,
            "day_off_reason": (
                holiday_name if is_holiday
                else ("Weekend" if is_weekend else None)
            ),
        }