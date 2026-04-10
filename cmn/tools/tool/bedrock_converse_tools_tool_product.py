import pandas as pd
import duckdb
import logging
from cmn.tools.tool.bedrock_converse_tools_tool import AbstractBedrockConverseTool

logger = logging.getLogger(__name__)


# ── 1. Mock Data ──────────────────────────────────────────────────────────────

def _build_mock_products() -> tuple[pd.DataFrame, pd.DataFrame]:

    products = pd.DataFrame({
        "product_id":   [1, 2, 3, 4, 5, 6, 7, 8],
        "product_name": ["Laptop Pro", "Laptop Air", "Wireless Mouse",
                         "Mechanical Keyboard", "USB-C Hub", "Monitor 27\"",
                         "Webcam HD", "Headset Pro"],
        "category":     ["Electronics", "Electronics", "Accessories",
                         "Accessories", "Accessories", "Electronics",
                         "Accessories", "Accessories"],
        "unit_price":   [1299.99, 899.99, 49.99, 129.99, 39.99,
                         499.99, 89.99, 149.99],
        "stock":        [50, 80, 200, 150, 300, 40, 120, 90],
        "rating":       [4.8, 4.6, 4.3, 4.7, 4.1, 4.9, 4.2, 4.5],
        "launched_year":[2022, 2023, 2021, 2022, 2023, 2021, 2022, 2023],
    })

    product_colors = pd.DataFrame({
        "color_id":   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "product_id": [1, 1, 2, 2, 3, 3, 3, 4, 4, 5,  6,  7,  8,  8 ],
        "color_name": ["Space Gray", "Silver", "Space Gray", "White",
                       "Black", "White", "Red", "Black", "White",
                       "Gray", "Black", "Black", "Black", "White"],
        "hex_code":   ["#8E8E93", "#C7C7CC", "#8E8E93", "#FFFFFF",
                       "#000000", "#FFFFFF", "#FF3B30", "#000000", "#FFFFFF",
                       "#8E8E93", "#000000", "#000000", "#000000", "#FFFFFF"],
        "stock_qty":  [20, 30, 40, 40, 80, 70, 50, 90, 60,
                       300, 40, 120, 50, 40],
    })

    return products, product_colors


_PRODUCTS_DF, _PRODUCT_COLORS_DF = _build_mock_products()


# ── 2. Schema Description (fed to LLM so it can write SQL) ───────────────────

SCHEMA_DESCRIPTION = """
You have access to two tables:

TABLE: products
  - product_id    INTEGER  (primary key)
  - product_name  VARCHAR
  - category      VARCHAR  (values: 'Electronics', 'Accessories')
  - unit_price    DOUBLE
  - stock         INTEGER  (total stock across all colors)
  - rating        DOUBLE   (1.0 to 5.0)
  - launched_year INTEGER

TABLE: product_colors
  - color_id      INTEGER  (primary key)
  - product_id    INTEGER  (foreign key → products.product_id)
  - color_name    VARCHAR
  - hex_code      VARCHAR
  - stock_qty     INTEGER  (stock for this specific color)

RELATIONSHIPS:
  - product_colors.product_id → products.product_id

RULES:
  - Use DuckDB SQL syntax
  - Always qualify ambiguous column names with table name
  - Return only the SQL query, no markdown, no explanation
  - Do not use semicolons at the end
"""


# ── 3. Tool Class ─────────────────────────────────────────────────────────────

class ProductBedrockConverseTool(AbstractBedrockConverseTool):
    """
    True NL-to-SQL tool.
    The LLM writes the actual SQL query — not just parameters.
    DuckDB executes it against in-memory DataFrames.
    """

    def __init__(self):
        name = "product_query"
        definition = {
            "toolSpec": {
                "name": name,
                "description": (
                    "Query the product catalog database using SQL. "
                    "Use this tool to answer ANY question about products and their colors. "
                    "You write the SQL query based on the schema provided. "
                    f"\n\nSCHEMA:\n{SCHEMA_DESCRIPTION}"
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": (
                                    "A valid DuckDB SQL SELECT query to run "
                                    "against the products and product_colors tables. "
                                    "Example: SELECT * FROM products WHERE unit_price > 100"
                                ),
                            },
                            "question": {
                                "type": "string",
                                "description": "The original user question being answered.",
                            },
                        },
                        "required": ["sql", "question"],
                    }
                },
            }
        }
        super().__init__(name, definition)

    def summary(self) -> str:
        return (
            "product_query : queries product catalog, colors and pricing. "
            "Writes and executes SQL against products and product_colors tables. "
        )
    # ── Invoke ────────────────────────────────────────────────────────────────

    def invoke(self, params, tool_args: dict = None) -> dict:
        args     = tool_args or {}
        sql      = args.get("sql", "").strip()
        question = args.get("question", "")

        logger.info("ProductTool — question: %s", question)
        logger.info("ProductTool — sql: %s", sql)

        if not sql:
            return {"error": "No SQL query provided"}

        # Safety guard — only allow SELECT
        if not sql.upper().startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed"}

        try:
            con = duckdb.connect()
            con.register("products",       _PRODUCTS_DF)
            con.register("product_colors", _PRODUCT_COLORS_DF)

            df = con.execute(sql).df()

            return {
                "question":  question,
                "sql_used":  sql,
                "row_count": len(df),
                "columns":   list(df.columns),
                "data":      df.to_dict(orient="records"),
            }

        except Exception as e:
            logger.error("ProductTool SQL error: %s | sql: %s", e, sql)
            return {
                "error":   str(e),
                "sql_used": sql,
                "hint":    "Check column names and table names against the schema",
            }