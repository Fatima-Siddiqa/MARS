"""Tunable constants for the Sector & Fundamentals agent, in one place."""

# A market price older than this (calendar days) is not accepted as "the price on the date".
MAX_PRICE_AGE_DAYS = 7
# A balance sheet older than this (days after its period end) is reported as stale.
STALE_STATEMENT_DAYS = 200

UNIT_TO_MILLIONS = {"PKR": 1e-6, "PKR_thousands": 1e-3, "PKR_millions": 1.0}

# Flow items are amounts over a period (summed into trailing twelve months).
FLOW_ITEMS = {
    "revenue",
    "markup_earned",
    "total_income",
    "net_interest_income",
    "operating_expenses",
    "net_income",
}
# Stock items are balances at a date (taken from the latest balance sheet).
STOCK_ITEMS = {
    "total_assets",
    "total_liabilities",
    "total_equity",
    "share_capital",
    "advances",
    "deposits",
    "investments",
    "cash_and_equivalents",
    "total_borrowings",
}
