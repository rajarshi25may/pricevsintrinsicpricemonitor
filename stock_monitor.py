#!/usr/bin/env python3

import sys
import os
import logging
import re
import smtplib
import requests
import feedparser
import pandas as pd
import yfinance as yf

from html import escape
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# SETTINGS FROM ENVIRONMENT / GITHUB SECRETS
# ============================================================

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Your existing Google Sheet details
SHEET_ID = os.getenv("SHEET_ID", "1_lQwmuBIzjg3kmc43sxML9vFt9qjyvOLmBGYW12LDLM")
SHEET_GID = os.getenv("SHEET_GID", "1600436104")

NEWS_ARTICLES_TO_FETCH = int(os.getenv("NEWS_ARTICLES_TO_FETCH", "5"))

# deepseek-chat is being deprecated, so use the newer model name
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

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
# BASIC VALIDATION
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


# ============================================================
# LOAD GOOGLE SHEET
# ============================================================

def load_sheet():
    log.info("Loading Google Sheet")

    try:
        df = pd.read_csv(SHEET_CSV_URL)
    except Exception as e:
        raise RuntimeError(f"Could not read Google Sheet CSV: {e}")

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

        notes_parts = []

        for i in range(4, min(7, len(row))):
            text = clean_text(row.iloc[i])
            if text:
                notes_parts.append(text)

        notes = " ".join(notes_parts)

        stocks.append({
            "name": name,
            "ticker": ticker,
            "low": float(low),
            "high": float(high),
            "notes": notes,
        })

    log.info("Loaded %s valid stock rows", len(stocks))
    return stocks


# ============================================================
# PRICE FETCH
# ============================================================

def ticker_candidates(ticker):
    """
    If the sheet already has .NS or .BO, use it directly.
    If it has only the NSE symbol, try original, then .NS, then .BO.
    """

    ticker = ticker.strip()

    candidates = [ticker]

    if "." not in ticker and not ticker.startswith("^"):
        candidates.append(f"{ticker}.NS")
        candidates.append(f"{ticker}.BO")

    # remove duplicates while preserving order
    return list(dict.fromkeys(candidates))


def get_price(ticker):
    for candidate in ticker_candidates(ticker):
        try:
            log.info("Fetching price for %s", candidate)

            t = yf.Ticker(candidate)

            price = None

            try:
                fast_info = t.fast_info
                price = fast_info.get("last_price")
            except Exception:
                price = None

            if price is not None and float(price) > 0:
                return float(price), candidate

            hist = t.history(period="5d", interval="1d")

            if hist is not None and not hist.empty:
                close_price = hist["Close"].dropna().iloc[-1]
                if close_price and float(close_price) > 0:
                    return float(close_price), candidate

        except Exception as e:
            log.warning("Price fetch failed for %s: %s", candidate, e)

    return None, ticker


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
    except Exception as e:
        log.warning("News fetch failed for %s: %s", stock_name, e)
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

        articles.append({
            "title": entry.get("title", "").strip(),
            "link": entry.get("link", "").strip(),
            "source": source_title,
            "published": entry.get("published", "").strip(),
            "snippet": snippet,
        })

    return articles


# ============================================================
# DEEPSEEK ANALYSIS
# ============================================================

def analyse(name, price, low, high, articles, notes):
    news_text = "\n\n".join(
        f"Title: {a['title']}\n"
        f"Source: {a['source']}\n"
        f"Published: {a['published']}\n"
        f"Snippet: {a['snippet']}"
        for a in articles
    )

    if not news_text:
        news_text = "No recent news articles were fetched."

    prompt = f"""
You are analysing an Indian listed stock for a long-term value investor.

Stock: {name}
Current price: ₹{price}
User's intrinsic value range (high to low): ₹{high} to ₹{low}

Investor notes:
{notes}

Recent news fetched:
{news_text}

Task:
1. Explain the most likely reason the stock is rising or falling.
2. Give a bull interpretation.
3. Give a bear interpretation.
4. Mention analyst/institutional fair value or target price only if it appears in the provided news text.
5. If analyst target price is not present in the news text, clearly say: "No external analyst target price found in fetched news."
6. Do not invent numbers.
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
                "temperature": 0.3,
                "max_tokens": 900,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        log.exception("DeepSeek analysis failed for %s", name)
        return f"AI analysis failed: {e}"


# ============================================================
# EMAIL
# ============================================================

def build_news_html(articles):
    if not articles:
        return "<p>No recent news fetched.</p>"

    items = []

    for article in articles:
        title = escape(article.get("title", "Untitled"))
        link = escape(article.get("link", ""))
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


def send_email(alerts, total_stocks_scanned, failed_prices):
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
        "%d-%m-%Y %I:%M %p IST"
    )

    if alerts:
        subject = f"{len(alerts)} stocks below intrinsic value"
    else:
        subject = "Daily stock monitor: no stocks below intrinsic value"

    if alerts:
        blocks = ""

        for alert in alerts:
            name = escape(alert["name"])
            original_ticker = escape(alert["ticker"])
            yahoo_ticker = escape(alert["yahoo_ticker"])

            notes = escape(alert["notes"]).replace("\n", "<br>")
            analysis = escape(alert["analysis"]).replace("\n", "<br>")
            news_html = build_news_html(alert["articles"])

            blocks += f"""
            <h2>{name} ({original_ticker})</h2>

            <p>
            <b>Yahoo ticker used:</b> {yahoo_ticker}<br>
            <b>Current price:</b> ₹{alert["price"]:.2f}<br>
            <b>Your intrinsic value range (high to low):</b> ₹{alert["high"]:.2f} - ₹{alert["low"]:.2f}
            </p>

            <p>
            <b>Your Notes</b><br>
            {notes if notes else "No notes found."}
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
        <p>No stock in your sheet is currently below your stated higher intrinsic value.</p>
        """

    failed_html = ""

    if failed_prices:
        failed_html = "<h3>Price fetch failures</h3><ul>"
        for item in failed_prices:
            failed_html += f"<li>{escape(item)}</li>"
        failed_html += "</ul>"

    html = f"""
    <html>
    <body>
        <h1>Daily Stock Monitor</h1>

        <p>
        <b>Run time:</b> {escape(now_ist)}<br>
        <b>Total stocks scanned:</b> {total_stocks_scanned}<br>
        <b>Alerts:</b> {len(alerts)}
        </p>

        {blocks}

        {failed_html}

        <p>
        <small>
        This is an automated GitHub Actions email.
        Prices are fetched from Yahoo Finance through yfinance.
        News is fetched from Google News RSS.
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

    except Exception as e:
        raise RuntimeError(f"Email sending failed: {e}")


# ============================================================
# SCAN
# ============================================================

def run_scan():
    require_env_vars()

    stocks = load_sheet()

    alerts = []
    failed_prices = []

    for stock in stocks:
        name = stock["name"]
        ticker = stock["ticker"]
        low = stock["low"]
        high = stock["high"]

        price, yahoo_ticker = get_price(ticker)

        if price is None:
            msg = f"{name} ({ticker})"
            failed_prices.append(msg)
            log.warning("Skipping %s because price could not be fetched", msg)
            continue

        log.info(
            "%s (%s): price %.2f, intrinsic high %.2f",
            name,
            yahoo_ticker,
            price,
            high,
        )

        if price < high:
            log.info("Alert triggered for %s", name)

            articles = fetch_news(name)

            analysis = analyse(
                name=name,
                price=price,
                low=low,
                high=high,
                articles=articles,
                notes=stock["notes"],
            )

            alerts.append({
                "name": name,
                "ticker": ticker,
                "yahoo_ticker": yahoo_ticker,
                "price": price,
                "low": low,
                "high": high,
                "notes": stock["notes"],
                "analysis": analysis,
                "articles": articles,
            })

    send_email(
        alerts=alerts,
        total_stocks_scanned=len(stocks),
        failed_prices=failed_prices,
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    try:
        run_scan()
    except Exception as e:
        log.exception("Stock monitor failed")
        raise
