import pandas as pd
import duckdb
import json
import logging
from cmn.bedrock_converse_tools import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)

def _safe_int(value) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


# ── 1. Mock Sales Data ────────────────────────────────────────────────────────

def _build_mock_sales() -> pd.DataFrame:
    """
    Generate 2 years of monthly sales data (2023 & 2024).
    Simulates a retail business with seasonal patterns and a mid-2024 dip.
    """

    records = [
        # year  month  revenue     units  returns  cogs       region      category
        # ── 2023 ──────────────────────────────────────────────────────────────
        (2023,  1,     120_000,    400,   20,      72_000,   "North",    "Electronics"),
        (2023,  1,      85_000,    600,   15,      42_500,   "South",    "Accessories"),
        (2023,  2,     115_000,    380,   18,      69_000,   "North",    "Electronics"),
        (2023,  2,      78_000,    550,   12,      39_000,   "South",    "Accessories"),
        (2023,  3,     130_000,    420,   22,      78_000,   "North",    "Electronics"),
        (2023,  3,      92_000,    640,   18,      46_000,   "South",    "Accessories"),
        (2023,  4,     140_000,    460,   25,      84_000,   "North",    "Electronics"),
        (2023,  4,      98_000,    680,   20,      49_000,   "South",    "Accessories"),
        (2023,  5,     155_000,    500,   28,      93_000,   "North",    "Electronics"),
        (2023,  5,     105_000,    720,   22,      52_500,   "South",    "Accessories"),
        (2023,  6,     160_000,    520,   30,      96_000,   "North",    "Electronics"),
        (2023,  6,     110_000,    750,   25,      55_000,   "South",    "Accessories"),
        (2023,  7,     158_000,    510,   29,      94_800,   "North",    "Electronics"),
        (2023,  7,     108_000,    740,   24,      54_000,   "South",    "Accessories"),
        (2023,  8,     162_000,    525,   31,      97_200,   "North",    "Electronics"),
        (2023,  8,     112_000,    760,   26,      56_000,   "South",    "Accessories"),
        (2023,  9,     170_000,    550,   33,     102_000,   "North",    "Electronics"),
        (2023,  9,     118_000,    800,   28,      59_000,   "South",    "Accessories"),
        (2023, 10,     180_000,    580,   35,     108_000,   "North",    "Electronics"),
        (2023, 10,     125_000,    850,   30,      62_500,   "South",    "Accessories"),
        (2023, 11,     210_000,    680,   40,     126_000,   "North",    "Electronics"),  # holiday
        (2023, 11,     148_000,   1000,   38,      74_000,   "South",    "Accessories"),  # holiday
        (2023, 12,     250_000,    800,   50,     150_000,   "North",    "Electronics"),  # peak
        (2023, 12,     175_000,   1200,   45,      87_500,   "South",    "Accessories"),  # peak

        # ── 2024 ──────────────────────────────────────────────────────────────
        (2024,  1,     125_000,    415,   21,      75_000,   "North",    "Electronics"),  # slight growth
        (2024,  1,      88_000,    620,   16,      44_000,   "South",    "Accessories"),
        (2024,  2,     118_000,    390,   19,      70_800,   "North",    "Electronics"),
        (2024,  2,      80_000,    560,   13,      40_000,   "South",    "Accessories"),
        (2024,  3,     135_000,    435,   23,      81_000,   "North",    "Electronics"),
        (2024,  3,      95_000,    660,   19,      47_500,   "South",    "Accessories"),
        (2024,  4,     132_000,    440,   26,      79_200,   "North",    "Electronics"),  # ← dip starts
        (2024,  4,      90_000,    640,   22,      45_000,   "South",    "Accessories"),  # ← dip starts
        (2024,  5,     128_000,    420,   30,      76_800,   "North",    "Electronics"),  # ← dip
        (2024,  5,      88_000,    610,   28,      44_000,   "South",    "Accessories"),  # ← dip
        (2024,  6,     122_000,    400,   35,      73_200,   "North",    "Electronics"),  # ← worst month
        (2024,  6,      82_000,    580,   32,      41_000,   "South",    "Accessories"),  # ← worst month
        (2024,  7,     145_000,    475,   27,      87_000,   "North",    "Electronics"),  # recovery
        (2024,  7,     100_000,    700,   22,      50_000,   "South",    "Accessories"),  # recovery
        (2024,  8,     155_000,    505,   29,      93_000,   "North",    "Electronics"),
        (2024,  8,     108_000,    740,   25,      54_000,   "South",    "Accessories"),
        (2024,  9,     165_000,    535,   31,      99_000,   "North",    "Electronics"),
        (2024,  9,     115_000,    780,   27,      57_500,   "South",    "Accessories"),
        (2024, 10,     175_000,    565,   34,     105_000,   "North",    "Electronics"),
        (2024, 10,     122_000,    830,   29,      61_000,   "South",    "Accessories"),
        (2024, 11,     205_000,    665,   39,     123_000,   "North",    "Electronics"),
        (2024, 11,     144_000,    980,   37,      72_000,   "South",    "Accessories"),
        (2024, 12,     245_000,    785,   48,     147_000,   "North",    "Electronics"),
        (2024, 12,     170_000,   1175,   44,      85_000,   "South",    "Accessories"),
    ]

    df = pd.DataFrame(records, columns=[
        "year", "month", "revenue", "units_sold",
        "returns", "cogs", "region", "category"
    ])

    # Derived columns
    # df["gross_profit"]    = df["revenue"] - df["cogs"]
    # df["profit_margin"]   = (df["gross_profit"] / df["revenue"] * 100).round(2)
    # df["net_revenue"]     = df["revenue"] - (df["returns"] * (df["revenue"] / df["units_sold"]))
    # df["month_name"]      = pd.to_datetime(df["month"], format="%m").dt.strftime("%B")

    df["gross_profit"]  = (df["revenue"] - df["cogs"]).fillna(0)
    df["profit_margin"] = (df["gross_profit"] / df["revenue"] * 100).fillna(0).round(2)
    df["net_revenue"]   = (df["revenue"] - (df["returns"] * (df["revenue"] / df["units_sold"]))).fillna(0)
    df["month_name"]    = pd.to_datetime(df["month"], format="%m").dt.strftime("%B")

    df = df.fillna(0)   # ← catch-all at the end

    return df


# Singleton — built once, reused across all tool calls
_SALES_DF = _build_mock_sales()


# ── 2. Query Helpers ──────────────────────────────────────────────────────────

def _run_sql(sql: str) -> pd.DataFrame:
    """Execute SQL against the in-memory sales DataFrame via DuckDB."""
    con = duckdb.connect()
    con.register("sales", _SALES_DF)
    return con.execute(sql).df()


def _df_to_tool_result(df: pd.DataFrame) -> dict:
    """
    Convert DataFrame to a JSON-serialisable dict the LLM can reason about.
    Keeps it compact — summary stats + row records.
    """
    return {
        "row_count": len(df),
        "columns":   list(df.columns),
        "data":      df.to_dict(orient="records"),
    }


# ── 3. Tool Class ─────────────────────────────────────────────────────────────
@DeprecationWarning
class SalesBedrockConverseTool(AbstractBedrockConverseTool):
    """
    Provides two query modes:
      • get_monthly_sales   → all months for a given year (+ optional category/region filter)
      • get_sales_by_month  → single month detail broken down by region & category
    """

    def __init__(self):
        name = "sales_data"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Retrieves sales performance data from the database. "
                    "Use 'get_monthly_sales' to get a full year of monthly data. "
                    "Use 'get_sales_by_month' to drill into a specific month. "
                    "Always call this tool for BOTH years when doing year-over-year comparisons."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query_type": {
                                "type": "string",
                                "enum": ["get_monthly_sales", "get_sales_by_month"],
                                "description": (
                                    "get_monthly_sales: returns all 12 months for a year, "
                                    "aggregated across regions and categories. "
                                    "get_sales_by_month: returns a single month broken down "
                                    "by region and category."
                                ),
                            },
                            "year": {
                                "type": "integer",
                                "description": "The year to query. Example: 2024",
                            },
                            "month": {
                                "type": "integer",
                                "description": (
                                    "Month number 1-12. "
                                    "Required only for get_sales_by_month."
                                ),
                            },
                            "category": {
                                "type": "string",
                                "enum": ["Electronics", "Accessories", "all"],
                                "description": (
                                    "Filter by product category. "
                                    "Defaults to 'all' if not specified."
                                ),
                            },
                            "region": {
                                "type": "string",
                                "enum": ["North", "South", "all"],
                                "description": (
                                    "Filter by region. "
                                    "Defaults to 'all' if not specified."
                                ),
                            },
                        },
                        "required": ["query_type", "year"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        return (
            "sales_data : fetches sales figures, monthly data and yearly summaries. "
            "For year-over-year comparisons call once per year. "
        )

    def invoke(self, params, tool_args: dict = None) -> dict:
        args       = tool_args or {}
        query_type = args.get("query_type")
        year       = args.get("year")
        month      = args.get("month")
        category   = args.get("category", "all")
        region     = args.get("region",   "all")

        logger.info(
            "SalesTool invoked: query_type=%s year=%s month=%s "
            "category=%s region=%s",
            query_type, year, month, category, region,
        )

        if query_type == "get_monthly_sales":
            return self._get_monthly_sales(year, category, region)

        if query_type == "get_sales_by_month":
            if month is None:
                return {"error": "month is required for get_sales_by_month"}
            return self._get_sales_by_month(year, month, category, region)

        return {"error": f"Unknown query_type: {query_type}"}

    # ── Query Implementations ─────────────────────────────────────────────────

    def _get_monthly_sales(
        self,
        year:     int,
        category: str = "all",
        region:   str = "all",
    ) -> dict:

        filters = ["year = ?"]
        params  = [year]

        if category != "all":
            filters.append("category = ?")
            params.append(category)
        if region != "all":
            filters.append("region = ?")
            params.append(region)

        where = " AND ".join(filters)

        sql = f"""
            SELECT
                month,
                month_name,
                SUM(revenue)       AS revenue,
                SUM(units_sold)    AS units_sold,
                SUM(returns)       AS returns,
                SUM(gross_profit)  AS gross_profit,
                ROUND(AVG(profit_margin), 2) AS profit_margin_pct,
                ROUND(SUM(net_revenue), 2)   AS net_revenue
            FROM sales
            WHERE {where}
            GROUP BY month, month_name
            ORDER BY month
        """

        con = duckdb.connect()
        con.register("sales", _SALES_DF)
        df = con.execute(sql, params).df()

        # ── Guard: no data found ──────────────────────────────────────────────
        if df.empty:
            return {
                "error":    f"No sales data found for year={year} category={category} region={region}",
                "row_count": 0,
                "data":     [],
            }
        df = df.fillna(0)

        df["ytd_revenue"] = df["revenue"].cumsum()

        result = _df_to_tool_result(df)
        result["year"]     = year
        result["category"] = category
        result["region"]   = region
        result["summary"] = {
            "total_revenue":      _safe_int(df["revenue"].sum()),
            "total_units":        _safe_int(df["units_sold"].sum()),
            "total_returns":      _safe_int(df["returns"].sum()),
            "avg_monthly_rev":    _safe_int(df["revenue"].mean()),
            "best_month":         df.loc[df["revenue"].idxmax(), "month_name"] if not df["revenue"].isna().all() else "N/A",
            "worst_month":        df.loc[df["revenue"].idxmin(), "month_name"] if not df["revenue"].isna().all() else "N/A",
            "total_gross_profit": _safe_int(df["gross_profit"].sum()),
        }

        return result

    def _get_sales_by_month(
        self,
        year:     int,
        month:    int,
        category: str = "all",
        region:   str = "all",
    ) -> dict:
        """
        Returns a single month broken down by region × category.
        """
        filters = ["year = ?", "month = ?"]
        params  = [year, month]

        if category != "all":
            filters.append("category = ?")
            params.append(category)
        if region != "all":
            filters.append("region = ?")
            params.append(region)

        where = " AND ".join(filters)

        sql = f"""
            SELECT
                year,
                month,
                month_name,
                region,
                category,
                revenue,
                units_sold,
                returns,
                gross_profit,
                profit_margin,
                ROUND(net_revenue, 2) AS net_revenue
            FROM sales
            WHERE {where}
            ORDER BY region, category
        """

        con = duckdb.connect()
        con.register("sales", _SALES_DF)
        print(sql)
        df = con.execute(sql, params).df()

        # ── Guard: no data found ──────────────────────────────────────────────
        if df.empty:
            return {
                "error":     f"No data found for year={year} month={month} category={category} region={region}",
                "row_count": 0,
                "data":      [],
            }

        result = _df_to_tool_result(df)
        result["year"]  = year
        result["month"] = month
        result["summary"] = {
            "total_revenue":        _safe_int(df["revenue"].sum()),
            "total_units":          _safe_int(df["units_sold"].sum()),
            "total_returns":        _safe_int(df["returns"].sum()),
            "total_gross_profit":   _safe_int(df["gross_profit"].sum()),
        }
        return result