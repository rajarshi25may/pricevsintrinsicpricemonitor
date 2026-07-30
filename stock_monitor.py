#!/usr/bin/env python3

import logging
import os
import re
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from zoneinfo import ZoneInfo

import feedparser
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# SETTINGS FROM ENVIRONMENT / GITHUB SECRETS
# ============================================================

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Existing Google Sheet details
SHEET_ID = os.getenv(
    "SHEET_ID",
    "1_lQwmuBIzjg3kmc43sxML9vFt9qjyvOLmBGYW12LDLM",
)
SHEET_GID = os.getenv("SHEET_GID", "1600436104")

NEWS_ARTICLES_TO_FETCH = int(os.getenv("NEWS_ARTICLES_TO_FETCH", "5"))

# Yahoo generally reports recommendation snapshots as 0m, -1m, -2m, -3m.
# 0m means the latest/current recommendation summary.
ANALYST_RECOMMENDATION_PERIOD = os.getenv(
    "ANALYST_RECOMMENDATION_PERIOD",
    "0m",
)

# DeepSeek V4 Pro with thinking mode enabled.
# The legacy deepseek-reasoner alias was retired on 24 July 2026.
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEEPSEEK_REASONING_EFFORT = os.getenv(
    "DEEPSEEK_REASONING_EFFORT",
    "high",
)
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "4000"))
DEEPSEEK_TIMEOUT_SECONDS = int(
    os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "180")
)

SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
    f"/export?format=csv&gid={SHEET_GID}"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("stock_monitor.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

log = logging.getLogger(__name__)


# ============================================================
# BASIC VALIDATION AND VALUE CLEANING
# ============================================================

def require_env_vars():
    required = {
        "EMAIL_SENDER": EMAIL_SENDER,
        "EMAIL_PASSWORD": EMAIL_PASSWORD,
        "EMAIL_RECEIVER": EMAIL_RECEIVER,
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
    }

    missing = [key for key, value in required.items() if not value]

    if missing:
        raise RuntimeError(
            "Missing environment variables / GitHub secrets: "
            + ", ".join(missing)
        )


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {"nan", "none", "null"}:
        return ""

    return value


def safe_float(value):
    """Convert a value to a finite float; otherwise return None."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(number):
        return None

    return number


def safe_int(value):
    """Convert Yahoo/pandas numeric output to a non-negative integer."""
    number = safe_float(value)

    if number is None:
        return 0

    return max(0, int(round(number)))


def first_present(mapping, *keys):
    """Return the first usable numeric value from alternative key names."""
    if not isinstance(mapping, dict):
        return None

    for key in keys:
        if key in mapping:
            value = safe_float(mapping.get(key))
            if value is not None:
                return value

    return None


# ============================================================
# LOAD GOOGLE SHEET
# ============================================================

def load_sheet():
    log.info("Loading Google Sheet")

    try:
        df = pd.read_csv(SHEET_CSV_URL)
    except Exception as exc:
        raise RuntimeError(f"Could not read Google Sheet CSV: {exc}") from exc

    stocks = []

    for _, row in df.iterrows():
        if len(row) < 4:
            continue

        name = clean_text(row.iloc[0])
        ticker = clean_text(row.iloc[1])

        if not name or not ticker:
            continue

        low = pd.to_numeric(row.iloc[2], errors="coerce")
        high = pd.to_numeric(row.iloc[3], errors="coerce")

        if pd.isna(low) or pd.isna(high):
            continue

        low = float(low)
        high = float(high)

        # Guard against accidentally reversed valuation columns.
        if low > high:
            log.warning(
                "%s (%s) has low %.2f greater than high %.2f; swapping them",
                name,
                ticker,
                low,
                high,
            )
            low, high = high, low

        notes_parts = []

        for i in range(4, min(7, len(row))):
            text = clean_text(row.iloc[i])
            if text:
                notes_parts.append(text)

        notes = " ".join(notes_parts)

        stocks.append(
            {
                "name": name,
                "ticker": ticker,
                "low": low,
                "high": high,
                "notes": notes,
            }
        )

    log.info("Loaded %s valid stock rows", len(stocks))
    return stocks


# ============================================================
# YAHOO TICKER RESOLUTION, PRICE AND DAILY MOVE
# ============================================================

def ticker_candidates(ticker):
    """
    If a suffix is already supplied, use it directly.

    For an unsuffixed Indian symbol, try NSE first, then the original symbol,
    then BSE. NSE first avoids accidentally resolving a same-named foreign
    security when the Google Sheet contains an Indian NSE symbol.
    """
    ticker = ticker.strip()

    if "." in ticker or ticker.startswith("^"):
        return [ticker]

    candidates = [f"{ticker}.NS", ticker, f"{ticker}.BO"]

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(candidates))


def extract_price_data(ticker_object):
    """
    Fetch current/latest price and previous close.

    fast_info is attempted first. Price history is used both as a fallback and
    to fill the previous close when fast_info does not provide it.
    """
    price = None
    previous_close = None

    try:
        fast_info = ticker_object.fast_info
        price = safe_float(fast_info.get("last_price"))
        previous_close = safe_float(fast_info.get("previous_close"))
    except Exception as exc:
        log.debug("fast_info unavailable: %s", exc)

    history = None

    if price is None or previous_close is None:
        try:
            history = ticker_object.history(period="5d", interval="1d")
        except Exception as exc:
            log.debug("Price history unavailable: %s", exc)

    if history is not None and not history.empty and "Close" in history.columns:
        closes = history["Close"].dropna()

        if price is None and len(closes) >= 1:
            price = safe_float(closes.iloc[-1])

        if previous_close is None:
            if len(closes) >= 2:
                previous_close = safe_float(closes.iloc[-2])
            elif len(closes) == 1:
                previous_close = safe_float(closes.iloc[-1])

    if price is None or price <= 0:
        return None

    daily_change = None
    daily_change_pct = None

    if previous_close is not None and previous_close > 0:
        daily_change = price - previous_close
        daily_change_pct = (daily_change / previous_close) * 100

    return {
        "price": price,
        "previous_close": previous_close,
        "daily_change": daily_change,
        "daily_change_pct": daily_change_pct,
    }


# ============================================================
# ANALYST CONSENSUS AND TARGET RANGE
# ============================================================

def empty_analyst_data(error=""):
    return {
        "period": "",
        "strong_buy": 0,
        "buy_only": 0,
        "hold": 0,
        "sell_only": 0,
        "strong_sell": 0,
        "buy": 0,
        "sell": 0,
        "total": 0,
        "buy_pct": None,
        "hold_pct": None,
        "sell_pct": None,
        "target_current": None,
        "target_low": None,
        "target_high": None,
        "target_mean": None,
        "target_median": None,
        "mean_upside_pct": None,
        "low_upside_pct": None,
        "high_upside_pct": None,
        "available": False,
        "error": error,
    }


def normalise_recommendation_dataframe(raw):
    """Convert Yahoo recommendation output to a conventional DataFrame."""
    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, pd.DataFrame):
        return raw.copy()

    if isinstance(raw, dict):
        try:
            return pd.DataFrame(raw)
        except Exception:
            try:
                return pd.DataFrame.from_dict(raw, orient="index")
            except Exception:
                return pd.DataFrame()

    return pd.DataFrame()


def select_latest_recommendation_row(recommendations):
    """
    Select the requested period (normally 0m). If unavailable, use the first
    row returned by Yahoo, which is normally the newest summary.
    """
    if recommendations is None or recommendations.empty:
        return None, ""

    df = recommendations.copy()
    df.columns = [str(column).strip() for column in df.columns]

    if "period" in df.columns:
        periods = df["period"].astype(str).str.strip().str.lower()
        wanted = ANALYST_RECOMMENDATION_PERIOD.strip().lower()
        matches = df[periods == wanted]

        if not matches.empty:
            return matches.iloc[0], clean_text(matches.iloc[0].get("period"))

        return df.iloc[0], clean_text(df.iloc[0].get("period"))

    # Some versions may return period labels as the index.
    index_labels = [str(label).strip().lower() for label in df.index]
    wanted = ANALYST_RECOMMENDATION_PERIOD.strip().lower()

    if wanted in index_labels:
        position = index_labels.index(wanted)
        return df.iloc[position], str(df.index[position])

    return df.iloc[0], str(df.index[0]) if len(df.index) else ""


def fetch_analyst_data(ticker_object, current_price):
    """
    Fetch latest analyst rating counts and aggregate target-price statistics.

    Buy = Strong Buy + Buy
    Sell = Sell + Strong Sell

    Failure is non-fatal. Missing coverage is returned as an N/A data object.
    """
    result = empty_analyst_data()
    errors = []

    recommendations = pd.DataFrame()

    try:
        if hasattr(ticker_object, "get_recommendations_summary"):
            raw_recommendations = ticker_object.get_recommendations_summary()
        else:
            raw_recommendations = ticker_object.recommendations_summary

        recommendations = normalise_recommendation_dataframe(
            raw_recommendations
        )
    except Exception as exc:
        errors.append(f"recommendations: {exc}")
        log.warning("Analyst recommendation fetch failed: %s", exc)

    row, period = select_latest_recommendation_row(recommendations)

    if row is not None:
        # Use exact official names first, with a few defensive alternatives.
        strong_buy = safe_int(
            row.get("strongBuy", row.get("strong_buy", 0))
        )
        buy_only = safe_int(row.get("buy", 0))
        hold = safe_int(row.get("hold", 0))
        sell_only = safe_int(row.get("sell", 0))
        strong_sell = safe_int(
            row.get("strongSell", row.get("strong_sell", 0))
        )

        buy = strong_buy + buy_only
        sell = sell_only + strong_sell
        total = buy + hold + sell

        result.update(
            {
                "period": period,
                "strong_buy": strong_buy,
                "buy_only": buy_only,
                "hold": hold,
                "sell_only": sell_only,
                "strong_sell": strong_sell,
                "buy": buy,
                "sell": sell,
                "total": total,
                "buy_pct": (buy / total * 100) if total else None,
                "hold_pct": (hold / total * 100) if total else None,
                "sell_pct": (sell / total * 100) if total else None,
            }
        )

    targets = {}

    try:
        if hasattr(ticker_object, "get_analyst_price_targets"):
            raw_targets = ticker_object.get_analyst_price_targets()
        else:
            raw_targets = ticker_object.analyst_price_targets

        if isinstance(raw_targets, dict):
            targets = raw_targets
    except Exception as exc:
        errors.append(f"targets: {exc}")
        log.warning("Analyst target fetch failed: %s", exc)

    target_current = first_present(
        targets,
        "current",
        "currentPrice",
        "current_price",
    )
    target_low = first_present(
        targets,
        "low",
        "targetLowPrice",
        "target_low",
    )
    target_high = first_present(
        targets,
        "high",
        "targetHighPrice",
        "target_high",
    )
    target_mean = first_present(
        targets,
        "mean",
        "targetMeanPrice",
        "target_mean",
    )
    target_median = first_present(
        targets,
        "median",
        "targetMedianPrice",
        "target_median",
    )

    base_price = current_price or target_current

    def upside(target):
        if target is None or base_price is None or base_price <= 0:
            return None
        return ((target - base_price) / base_price) * 100

    result.update(
        {
            "target_current": target_current,
            "target_low": target_low,
            "target_high": target_high,
            "target_mean": target_mean,
            "target_median": target_median,
            "mean_upside_pct": upside(target_mean),
            "low_upside_pct": upside(target_low),
            "high_upside_pct": upside(target_high),
        }
    )

    result["available"] = bool(
        result["total"]
        or target_low is not None
        or target_high is not None
        or target_mean is not None
        or target_median is not None
    )
    result["error"] = "; ".join(errors)

    return result


def get_market_snapshot(ticker):
    """
    Resolve a ticker, fetch its price data, then fetch analyst data using the
    same resolved Yahoo Finance Ticker object.
    """
    candidate_errors = []

    for candidate in ticker_candidates(ticker):
        try:
            log.info("Fetching market data for %s", candidate)
            ticker_object = yf.Ticker(candidate)
            price_data = extract_price_data(ticker_object)

            if price_data is None:
                candidate_errors.append(f"{candidate}: no usable price")
                continue

            analyst_data = fetch_analyst_data(
                ticker_object=ticker_object,
                current_price=price_data["price"],
            )

            return {
                "yahoo_ticker": candidate,
                **price_data,
                "analyst": analyst_data,
            }

        except Exception as exc:
            candidate_errors.append(f"{candidate}: {exc}")
            log.warning("Market fetch failed for %s: %s", candidate, exc)

    return {
        "yahoo_ticker": ticker,
        "price": None,
        "previous_close": None,
        "daily_change": None,
        "daily_change_pct": None,
        "analyst": empty_analyst_data(
            error="; ".join(candidate_errors)
        ),
    }


# ============================================================
# NEWS FETCH
# ============================================================

def fetch_news(stock_name):
    log.info("Fetching news for %s", stock_name)

    query = requests.utils.quote(f"{stock_name} stock India")
    rss_url = (
        f"https://news.google.com/rss/search?q={query}"
        f"&hl=en-IN&gl=IN&ceid=IN:en"
    )

    try:
        response = requests.get(
            rss_url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 stock-monitor"},
        )
        response.raise_for_status()
    except Exception as exc:
        log.warning("News fetch failed for %s: %s", stock_name, exc)
        return []

    feed = feedparser.parse(response.content)
    articles = []

    for entry in feed.entries[:NEWS_ARTICLES_TO_FETCH]:
        raw_summary = entry.get("summary", "")
        snippet = re.sub("<.*?>", "", raw_summary).strip()

        source = entry.get("source", {})
        source_title = ""

        try:
            source_title = source.get("title", "")
        except Exception:
            source_title = ""

        articles.append(
            {
                "title": entry.get("title", "").strip(),
                "link": entry.get("link", "").strip(),
                "source": source_title,
                "published": entry.get("published", "").strip(),
                "snippet": snippet,
            }
        )

    return articles


# ============================================================
# DEEPSEEK ANALYSIS
# ============================================================

def analyst_data_for_prompt(analyst):
    if not analyst.get("available"):
        return "No structured Yahoo analyst consensus or target data available."

    lines = []

    if analyst.get("total", 0) > 0:
        lines.extend(
            [
                f"Recommendation period: {analyst.get('period') or 'latest'}",
                f"Buy: {analyst['buy']} "
                f"({analyst['buy_pct']:.1f}%)",
                f"Hold: {analyst['hold']} "
                f"({analyst['hold_pct']:.1f}%)",
                f"Sell: {analyst['sell']} "
                f"({analyst['sell_pct']:.1f}%)",
                f"Total recommendations counted: {analyst['total']}",
            ]
        )

    if analyst.get("target_low") is not None:
        lines.append(f"Lowest analyst target: ₹{analyst['target_low']:.2f}")

    if analyst.get("target_mean") is not None:
        lines.append(f"Mean analyst target: ₹{analyst['target_mean']:.2f}")

    if analyst.get("target_median") is not None:
        lines.append(
            f"Median analyst target: ₹{analyst['target_median']:.2f}"
        )

    if analyst.get("target_high") is not None:
        lines.append(f"Highest analyst target: ₹{analyst['target_high']:.2f}")

    if analyst.get("mean_upside_pct") is not None:
        lines.append(
            "Potential return to mean target: "
            f"{analyst['mean_upside_pct']:+.1f}%"
        )

    return "\n".join(lines) if lines else "No structured analyst data available."


def analyse(
    name,
    price,
    previous_close,
    daily_change_pct,
    low,
    high,
    articles,
    notes,
    analyst,
):
    news_text = "\n\n".join(
        f"Title: {article['title']}\n"
        f"Source: {article['source']}\n"
        f"Published: {article['published']}\n"
        f"Snippet: {article['snippet']}"
        for article in articles
    )

    if not news_text:
        news_text = "No recent news articles were fetched."

    if previous_close is not None and daily_change_pct is not None:
        movement_text = (
            f"Previous close: ₹{previous_close:.2f}\n"
            f"Latest movement: {daily_change_pct:+.2f}%"
        )
    else:
        movement_text = "Previous close and daily movement were unavailable."

    structured_analyst_text = analyst_data_for_prompt(analyst)

    prompt = f"""
You are analysing an Indian listed stock for a long-term value investor.

Stock: {name}
Current price: ₹{price:.2f}
{movement_text}
User's intrinsic value range: ₹{low:.2f} to ₹{high:.2f}

Investor notes:
{notes or 'No investor notes were supplied.'}

Structured Yahoo analyst consensus and target data:
{structured_analyst_text}

Recent news fetched:
{news_text}

Task:
1. Explain the most likely reason for the latest price movement, using the actual movement above and the supplied news. If the evidence is insufficient, say so plainly.
2. Give a bull interpretation.
3. Give a bear interpretation.
4. Compare current price with the user's intrinsic-value range.
5. Discuss analyst consensus and target range only from the structured Yahoo data above. Do not attribute the lowest target to Sell analysts or the highest target to Buy analysts because those links are not supplied.
6. Do not invent numbers, broker names, ratings or target prices.
7. Keep it concise but useful.
"""

    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "thinking": {"type": "enabled"},
                "reasoning_effort": DEEPSEEK_REASONING_EFFORT,
                "max_tokens": DEEPSEEK_MAX_TOKENS,
            },
            timeout=DEEPSEEK_TIMEOUT_SECONDS,
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    except Exception as exc:
        log.exception("DeepSeek analysis failed for %s", name)
        return f"AI analysis failed: {exc}"


# ============================================================
# EMAIL FORMATTING HELPERS
# ============================================================

def format_money(value):
    number = safe_float(value)
    return "N/A" if number is None else f"₹{number:,.2f}"


def format_percentage(value, include_sign=False):
    number = safe_float(value)

    if number is None:
        return "N/A"

    if include_sign:
        return f"{number:+.1f}%"

    return f"{number:.1f}%"


def count_with_percentage(count, percentage):
    if percentage is None:
        return "N/A" if not count else str(count)

    return f"{count} ({percentage:.1f}%)"


def format_target_range(analyst):
    low = analyst.get("target_low")
    high = analyst.get("target_high")

    if low is not None and high is not None:
        return f"{format_money(low)} – {format_money(high)}"

    if low is not None:
        return f"From {format_money(low)}"

    if high is not None:
        return f"Up to {format_money(high)}"

    return "N/A"


def status_label(row):
    if row.get("price") is None:
        return "PRICE FAILED"

    if row.get("is_alert"):
        return "ALERT"

    return "NO ALERT"


def status_cell_style(row):
    if row.get("price") is None:
        return "background:#f8d7da;color:#842029;font-weight:bold;"

    if row.get("is_alert"):
        return "background:#fff3cd;color:#664d03;font-weight:bold;"

    return "background:#d1e7dd;color:#0f5132;font-weight:bold;"


def build_portfolio_summary_table(portfolio_rows):
    """Create the all-stock analyst table shown at the top of the email."""
    if not portfolio_rows:
        return "<p>No valid stock rows were found in the Google Sheet.</p>"

    rows_html = []

    for row in portfolio_rows:
        analyst = row["analyst"]
        stock_name = escape(row["name"])
        ticker = escape(row["ticker"])
        yahoo_ticker = escape(row.get("yahoo_ticker", ""))

        daily_move = format_percentage(
            row.get("daily_change_pct"),
            include_sign=True,
        )

        own_range = (
            f"{format_money(row['low'])} – {format_money(row['high'])}"
        )

        rows_html.append(
            f"""
            <tr>
                <td>{stock_name}</td>
                <td>{ticker}<br><small>{yahoo_ticker}</small></td>
                <td style="text-align:right;">{format_money(row.get('price'))}</td>
                <td style="text-align:right;">{daily_move}</td>
                <td>{own_range}</td>
                <td style="text-align:center;">{
                    count_with_percentage(
                        analyst.get('buy', 0),
                        analyst.get('buy_pct'),
                    )
                }</td>
                <td style="text-align:center;">{
                    count_with_percentage(
                        analyst.get('hold', 0),
                        analyst.get('hold_pct'),
                    )
                }</td>
                <td style="text-align:center;">{
                    count_with_percentage(
                        analyst.get('sell', 0),
                        analyst.get('sell_pct'),
                    )
                }</td>
                <td style="text-align:center;">{
                    analyst.get('total', 0) if analyst.get('total', 0) else 'N/A'
                }</td>
                <td>{format_target_range(analyst)}</td>
                <td style="text-align:right;">{
                    format_money(analyst.get('target_mean'))
                }</td>
                <td style="text-align:right;">{
                    format_money(analyst.get('target_median'))
                }</td>
                <td style="text-align:right;">{
                    format_percentage(
                        analyst.get('mean_upside_pct'),
                        include_sign=True,
                    )
                }</td>
                <td style="{status_cell_style(row)}">{status_label(row)}</td>
            </tr>
            """
        )

    return f"""
    <h2>Portfolio-wide Analyst Consensus</h2>

    <p>
        This table covers every valid stock in the Google Sheet.
        <b>Buy</b> combines Strong Buy + Buy. <b>Sell</b> combines
        Sell + Strong Sell. The analyst target range is Yahoo's overall
        low-to-high target range; it is not a separate Buy range and Sell range.
    </p>

    <div style="overflow-x:auto;">
        <table style="border-collapse:collapse;width:100%;font-size:13px;">
            <thead>
                <tr style="background:#e9ecef;">
                    <th style="border:1px solid #999;padding:7px;">Stock</th>
                    <th style="border:1px solid #999;padding:7px;">Ticker</th>
                    <th style="border:1px solid #999;padding:7px;">Current</th>
                    <th style="border:1px solid #999;padding:7px;">1-day move</th>
                    <th style="border:1px solid #999;padding:7px;">Your value range</th>
                    <th style="border:1px solid #999;padding:7px;">Buy</th>
                    <th style="border:1px solid #999;padding:7px;">Hold</th>
                    <th style="border:1px solid #999;padding:7px;">Sell</th>
                    <th style="border:1px solid #999;padding:7px;">Total</th>
                    <th style="border:1px solid #999;padding:7px;">Analyst target range</th>
                    <th style="border:1px solid #999;padding:7px;">Mean target</th>
                    <th style="border:1px solid #999;padding:7px;">Median target</th>
                    <th style="border:1px solid #999;padding:7px;">To mean</th>
                    <th style="border:1px solid #999;padding:7px;">Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """.replace(
        "<td>",
        '<td style="border:1px solid #999;padding:7px;vertical-align:top;">',
    ).replace(
        '<td style="text-align:',
        '<td style="border:1px solid #999;padding:7px;vertical-align:top;text-align:',
    ).replace(
        '<td style="background:',
        '<td style="border:1px solid #999;padding:7px;vertical-align:top;background:',
    )


def build_news_html(articles):
    if not articles:
        return "<p>No recent news fetched.</p>"

    items = []

    for article in articles:
        title = escape(article.get("title", "Untitled"))
        link = escape(article.get("link", ""), quote=True)
        source = escape(article.get("source", ""))
        published = escape(article.get("published", ""))

        if link:
            item = f"<li><a href='{link}'>{title}</a>"
        else:
            item = f"<li>{title}"

        if source or published:
            item += f"<br><small>{source} {published}</small>"

        item += "</li>"
        items.append(item)

    return "<ul>" + "\n".join(items) + "</ul>"


def build_analyst_detail_html(analyst):
    if not analyst.get("available"):
        return "<p>No analyst consensus or target data available from Yahoo Finance.</p>"

    rating_html = ""

    if analyst.get("total", 0) > 0:
        rating_html = f"""
        <p>
            <b>Buy:</b> {
                count_with_percentage(
                    analyst['buy'],
                    analyst['buy_pct'],
                )
            }<br>
            <b>Hold:</b> {
                count_with_percentage(
                    analyst['hold'],
                    analyst['hold_pct'],
                )
            }<br>
            <b>Sell:</b> {
                count_with_percentage(
                    analyst['sell'],
                    analyst['sell_pct'],
                )
            }<br>
            <b>Total recommendations:</b> {analyst['total']}<br>
            <small>
                Detailed: Strong Buy {analyst['strong_buy']},
                Buy {analyst['buy_only']}, Hold {analyst['hold']},
                Sell {analyst['sell_only']},
                Strong Sell {analyst['strong_sell']}.
            </small>
        </p>
        """

    targets_html = f"""
    <p>
        <b>Overall analyst target range:</b> {format_target_range(analyst)}<br>
        <b>Mean target:</b> {format_money(analyst.get('target_mean'))}<br>
        <b>Median target:</b> {format_money(analyst.get('target_median'))}<br>
        <b>Potential to mean target:</b> {
            format_percentage(
                analyst.get('mean_upside_pct'),
                include_sign=True,
            )
        }
    </p>
    """

    return rating_html + targets_html


# ============================================================
# EMAIL
# ============================================================

def send_email(alerts, portfolio_rows, failed_prices):
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
        "%d-%m-%Y %I:%M %p IST"
    )

    if alerts:
        subject = f"{len(alerts)} valuation alerts | analyst consensus report"
    else:
        subject = "Daily stock monitor | no valuation alerts"

    summary_table = build_portfolio_summary_table(portfolio_rows)

    if alerts:
        blocks = ""

        for alert in alerts:
            name = escape(alert["name"])
            original_ticker = escape(alert["ticker"])
            yahoo_ticker = escape(alert["yahoo_ticker"])

            notes = escape(alert["notes"]).replace("\n", "<br>")
            analysis = escape(alert["analysis"]).replace("\n", "<br>")
            news_html = build_news_html(alert["articles"])
            analyst_html = build_analyst_detail_html(alert["analyst"])

            daily_move = format_percentage(
                alert.get("daily_change_pct"),
                include_sign=True,
            )

            blocks += f"""
            <h2>{name} ({original_ticker})</h2>

            <p>
                <b>Yahoo ticker used:</b> {yahoo_ticker}<br>
                <b>Current price:</b> {format_money(alert['price'])}<br>
                <b>Previous close:</b> {
                    format_money(alert.get('previous_close'))
                }<br>
                <b>Latest movement:</b> {daily_move}<br>
                <b>Your intrinsic-value range:</b>
                {format_money(alert['low'])} – {format_money(alert['high'])}
            </p>

            <h3>Analyst Consensus and Target Range</h3>
            {analyst_html}

            <p>
                <b>Your Notes</b><br>
                {notes if notes else 'No notes found.'}
            </p>

            <p>
                <b>AI Analysis</b><br>
                {analysis}
            </p>

            <p><b>News</b></p>
            {news_html}

            <hr>
            """
    else:
        blocks = """
        <p>
            No stock in the sheet is currently below its stated higher
            intrinsic-value estimate.
        </p>
        """

    failed_html = ""

    if failed_prices:
        failed_html = "<h3>Price fetch failures</h3><ul>"
        for item in failed_prices:
            failed_html += f"<li>{escape(item)}</li>"
        failed_html += "</ul>"

    html = f"""
    <html>
    <body style="font-family:Arial,Helvetica,sans-serif;color:#212529;">
        <h1>Daily Stock Monitor</h1>

        <p>
            <b>Run time:</b> {escape(now_ist)}<br>
            <b>Total valid stocks scanned:</b> {len(portfolio_rows)}<br>
            <b>Valuation alerts:</b> {len(alerts)}
        </p>

        {summary_table}

        <hr>
        <h1>Detailed Valuation Alerts</h1>

        {blocks}

        {failed_html}

        <p>
            <small>
                This is an automated GitHub Actions email. Prices, analyst
                recommendation summaries and analyst target ranges are fetched
                from Yahoo Finance through yfinance. News is fetched from
                Google News RSS. Analyst coverage may be missing for smaller
                or lightly covered companies.
            </small>
        </p>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=60) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(
                EMAIL_SENDER,
                [EMAIL_RECEIVER],
                msg.as_string(),
            )

        log.info("Email sent successfully")

    except Exception as exc:
        raise RuntimeError(f"Email sending failed: {exc}") from exc


# ============================================================
# SCAN
# ============================================================

def run_scan():
    require_env_vars()

    stocks = load_sheet()
    alerts = []
    portfolio_rows = []
    failed_prices = []

    for stock in stocks:
        name = stock["name"]
        ticker = stock["ticker"]
        low = stock["low"]
        high = stock["high"]

        snapshot = get_market_snapshot(ticker)
        price = snapshot["price"]

        is_alert = price is not None and price < high

        portfolio_row = {
            **stock,
            **snapshot,
            "is_alert": is_alert,
        }
        portfolio_rows.append(portfolio_row)

        if price is None:
            message = f"{name} ({ticker})"
            failed_prices.append(message)
            log.warning(
                "Skipping detailed alert analysis for %s because price "
                "could not be fetched",
                message,
            )
            continue

        log.info(
            "%s (%s): price %.2f, intrinsic high %.2f, "
            "buy %s, hold %s, sell %s",
            name,
            snapshot["yahoo_ticker"],
            price,
            high,
            snapshot["analyst"].get("buy", 0),
            snapshot["analyst"].get("hold", 0),
            snapshot["analyst"].get("sell", 0),
        )

        if is_alert:
            log.info("Alert triggered for %s", name)
            articles = fetch_news(name)

            analysis = analyse(
                name=name,
                price=price,
                previous_close=snapshot.get("previous_close"),
                daily_change_pct=snapshot.get("daily_change_pct"),
                low=low,
                high=high,
                articles=articles,
                notes=stock["notes"],
                analyst=snapshot["analyst"],
            )

            alerts.append(
                {
                    **portfolio_row,
                    "analysis": analysis,
                    "articles": articles,
                }
            )

    send_email(
        alerts=alerts,
        portfolio_rows=portfolio_rows,
        failed_prices=failed_prices,
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    try:
        run_scan()
    except Exception:
        log.exception("Stock monitor failed")
        raise
