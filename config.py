import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB", "market_predictor"),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# ── Assets ────────────────────────────────────────────────────────────────────
ALL_ASSETS = ["BTC", "ETH", "GOLD", "OIL"]

# ── Prediction horizons (hourly resolution) ───────────────────────────────────
HORIZONS_HOURS = [1, 4]

# ── External API keys ─────────────────────────────────────────────────────────
FINNHUB_KEY  = os.getenv("FINNHUB_KEY",  "")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY", "")
