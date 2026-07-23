# -*- coding: utf-8 -*-
"""
HexaTrades — Fibonacci Golden Zone Tarayıcı
--------------------------------------------
Bu uygulama, ekte paylaşılan Pine Script v6 stratejisinin
(Advanced Fibonacci Golden Zone Strategy) AL / SAT mantığını Python'da
bire bir mühendislik yaklaşımıyla yeniden üretir ve bunu BIST / ABD
borsalarındaki hisseler üzerinde canlı olarak tarar.

Strateji mantığı (Pine Script'ten birebir taşınmıştır):
  1) Pivot Tespiti      -> ta.pivothigh / ta.pivotlow (left=pivotLen, right=confirmBars)
  2) ZigZag İnşası      -> Ardışık pivotlardan, ATR filtreli (zzDevAtr) bacaklar
  3) Golden Zone        -> Her yeni bacakta 0.5 - 0.618 retracement kutusu
  4) AL Sinyali         -> (a) Golden Zone'a değip yukarı reddeden yeşil mum (evBullRej)
                            (b) Yeni bir ZigZag dibi onaylanması (isZigZagLow)
  5) SAT Sinyali        -> Kapanışın, son ZigZag dibi - (invBufAtr × ATR)
                            seviyesindeki trailing stop'un altına düşmesi

Tarayıcı, geçmişte en az bir SAT (çıkış) yaşanmış ve şu anki (son kapanmış)
bar'da taze bir AL sinyali üretmiş hisseleri "TAM AL NOKTASI" olarak listeler.

NOT: Bu bir yeniden-mühendislik / adaptasyondur; Pine'daki görsel çizim
(line/box/label) unsurları taramaya dahil edilmemiştir, sadece AL/SAT
karar mantığı birebir uygulanmıştır. Pivot eşitliklerinde (tie-break)
küçük yaklaşıklıklar olabilir.
"""

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# =============================================================================
# 0. SAYFA / STİL KONFİGÜRASYONU
# =============================================================================
st.set_page_config(
    page_title="Fibonacci Golden Zone Tarayıcı",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3 { font-weight: 800; letter-spacing: -0.3px; color: #e5e7eb; }
        h1 { border-bottom: 1px solid #374151; padding-bottom: 12px; }
        .stButton>button {
            width: 100%; background-color: #d97706; color: white;
            font-weight: 600; letter-spacing: 0.5px; border-radius: 4px;
            border: 1px solid #b45309; text-transform: uppercase;
        }
        .stButton>button:hover { background-color: #f59e0b; border-color: #d97706; }
        .buy-badge {
            display:inline-block; padding:3px 10px; border-radius:12px;
            background:#065f46; color:#6ee7b7; font-weight:700; font-size:12px;
            letter-spacing:0.5px;
        }
        div[data-testid="stMetric"] { background:#111827; padding:10px; border-radius:6px; border:1px solid #1f2937; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Fibonacci Golden Zone Tarayıcı")
st.caption("HexaTrades Pine Script stratejisinin (v6) Python tarayıcı adaptasyonu — sadece 'TAM AL NOKTASI'ndaki hisseleri listeler.")

# =============================================================================
# 1. PİYASA VE ZAMAN DİLİMİ KONFİGÜRASYONU
# =============================================================================
MARKET_CONFIGS = {
    "🇹🇷 BIST (Türkiye)": {"tv_market": "turkey", "yf_suffix": ".IS", "tv_prefix": "BIST:"},
    "🇺🇸 ABD (NASDAQ / NYSE)": {"tv_market": "america", "yf_suffix": "", "tv_prefix": ""},
}

TIMEFRAME_CONFIGS = {
    "1 Saat (1H)":  {"yf_interval": "60m", "resample": None, "period": "730d", "tv_interval": "60"},
    "2 Saat (2H)":  {"yf_interval": "60m", "resample": "2h", "period": "730d", "tv_interval": "120"},
    "4 Saat (4H)":  {"yf_interval": "60m", "resample": "4h", "period": "730d", "tv_interval": "240"},
    "1 Gün (1D)":   {"yf_interval": "1d",  "resample": None, "period": "5y",  "tv_interval": "D"},
    "1 Hafta (1W)": {"yf_interval": "1wk", "resample": None, "period": "10y", "tv_interval": "W"},
}

MIN_BARS_REQUIRED = 80  # pivot + zigzag için makul minimum geçmiş

# =============================================================================
# 2. SEMBOL LİSTESİ (TradingView Scanner)
# =============================================================================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_market_symbols(tv_market: str, limit: int = 300):
    """TradingView public scanner API üzerinden piyasa değerine göre sıralı sembol listesi çeker."""
    url = f"https://scanner.tradingview.com/{tv_market}/scan"
    payload = {
        "filter": [{"left": "type", "operation": "in_range", "right": ["stock"]}],
        "options": {"lang": "en"},
        "markets": [tv_market],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "market_cap_basic"],
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
        "range": [0, limit],
    }
    try:
        resp = requests.post(url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if resp.status_code == 200:
            return [item["d"][0] for item in resp.json().get("data", [])]
    except Exception:
        pass
    return []


# =============================================================================
# 3. VERİ ÇEKME (Toplu / Cache'li)
# =============================================================================
def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_batch(tickers: tuple, yf_interval: str, period: str):
    try:
        data = yf.download(
            tickers=list(tickers),
            period=period,
            interval=yf_interval,
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False,
        )
        return data
    except Exception:
        return pd.DataFrame()


def extract_symbol_df(batch: pd.DataFrame, yf_ticker: str, single_ticker: bool) -> pd.DataFrame:
    try:
        if single_ticker:
            df = batch.copy()
        else:
            if not hasattr(batch.columns, "levels"):
                return None
            if yf_ticker not in batch.columns.levels[0]:
                return None
            df = batch[yf_ticker].copy()
        if df is None or df.empty:
            return None
        df.columns = [str(c).lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


# =============================================================================
# 4. STRATEJİ MOTORU (Pine Script mantığının Python karşılığı)
# =============================================================================
def wilder_atr(df: pd.DataFrame, length: int = 14) -> np.ndarray:
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.nanmax(
        np.vstack([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]),
        axis=0,
    )
    tr_series = pd.Series(tr)
    atr = tr_series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return atr.values


def detect_pivots(df: pd.DataFrame, left: int, right: int):
    """ta.pivothigh / ta.pivotlow eşleniği (vektörize).
    p konumundaki bar, [p-left, p+right] penceresinde high/low ekstremumu ise
    pivot olarak işaretlenir; değer, current-bar indeksine (p) yerleştirilir."""
    window = left + right + 1
    high, low = df["high"], df["low"]
    rm_high = high.rolling(window, min_periods=window).max()
    rm_low = low.rolling(window, min_periods=window).min()
    aligned_high_max = rm_high.shift(-right)
    aligned_low_min = rm_low.shift(-right)
    ph_val = high.where(high == aligned_high_max)
    pl_val = low.where(low == aligned_low_min)
    return ph_val.values, pl_val.values


def run_strategy(
    df: pd.DataFrame,
    left: int = 15,
    right: int = 5,
    golden_lower: float = 0.5,
    golden_upper: float = 0.618,
    inv_buf_atr: float = 0.3,
    zz_dev_atr: float = 1.5,
    touch_wick: bool = True,
    skip_late: bool = True,
):
    n = len(df)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    openp = df["open"].values.astype(float)

    atr = wilder_atr(df, 14)
    ph_val, pl_val = detect_pivots(df, left, right)

    zzP0 = zzP1 = None
    zzX0 = zzX1 = None
    zzD1 = 0
    zzHigh = zzPrevHigh = zzLow = zzPrevLow = None

    aBull = False
    aSet = False
    aAlive = False
    aRejected = False
    aHigh = aLow = None
    aBornBar = None

    trailing_stop = np.nan
    position = False

    long_entry = np.zeros(n, dtype=bool)
    long_exit = np.zeros(n, dtype=bool)
    entry_from_gz = np.zeros(n, dtype=bool)
    entry_from_zz = np.zeros(n, dtype=bool)
    zone_top_arr = np.full(n, np.nan)
    zone_bot_arr = np.full(n, np.nan)
    zone_bull_arr = np.full(n, np.nan)

    near_ratio = min(golden_lower, golden_upper)

    for i in range(n):
        newPH = ph_val[i] if i - right >= 0 else np.nan
        newPL = pl_val[i] if i - right >= 0 else np.nan
        usePH, usePL = newPH, newPL

        if not np.isnan(newPH) and not np.isnan(newPL):
            if zzP1 is None:
                usePH = np.nan
                usePL = np.nan
            else:
                dH = abs(newPH - zzP1)
                dL = abs(newPL - zzP1)
                if dH > dL:
                    usePL = np.nan
                elif dL > dH:
                    usePH = np.nan
                else:
                    usePH = np.nan
                    usePL = np.nan

        lastPHx = i - right if not np.isnan(usePH) else None
        lastPLx = i - right if not np.isnan(usePL) else None

        zzLegEvent = False
        isZigZagLow = False

        pivot_atr_idx = i - right
        pivotAtr = atr[pivot_atr_idx] if (0 <= pivot_atr_idx < n and not np.isnan(atr[pivot_atr_idx])) else 0.0
        zzMinLeg = 0.0 if zz_dev_atr == 0.0 else zz_dev_atr * pivotAtr

        if not np.isnan(usePH):
            if zzD1 == 1:
                if usePH > zzP1:
                    zzP1, zzX1 = usePH, lastPHx
                    zzHigh = usePH
                    zzLegEvent = True
            elif zzP1 is None:
                zzP1, zzX1, zzD1 = usePH, lastPHx, 1
                zzHigh = usePH
            elif abs(usePH - zzP1) >= zzMinLeg:
                zzP0, zzX0 = zzP1, zzX1
                zzP1, zzX1, zzD1 = usePH, lastPHx, 1
                zzPrevHigh = zzHigh
                zzHigh = usePH
                zzLegEvent = True

        if not np.isnan(usePL):
            if zzD1 == -1:
                if usePL < zzP1:
                    zzP1, zzX1 = usePL, lastPLx
                    zzLow = usePL
                    zzLegEvent = True
                    isZigZagLow = True
            elif zzP1 is None:
                zzP1, zzX1, zzD1 = usePL, lastPLx, -1
                zzLow = usePL
                isZigZagLow = True
            elif abs(usePL - zzP1) >= zzMinLeg:
                zzP0, zzX0 = zzP1, zzX1
                zzP1, zzX1, zzD1 = usePL, lastPLx, -1
                zzPrevLow = zzLow
                zzLow = usePL
                zzLegEvent = True
                isZigZagLow = True

        # Trailing stop, her yeni ZigZag dibinde güncellenir
        if isZigZagLow:
            atr_now = atr[i] if not np.isnan(atr[i]) else 0.0
            trailing_stop = zzLow - inv_buf_atr * atr_now

        validLeg = (zzD1 != 0) and (zzP0 is not None) and (zzP1 is not None) and (zzP0 != zzP1)
        legBull = zzD1 == 1
        legHigh = max(zzP0, zzP1) if validLeg else None
        legLow = min(zzP0, zzP1) if validLeg else None

        candidateEvent = zzLegEvent and validLeg and (legHigh is not None) and (legHigh > legLow)

        if candidateEvent:
            dirBull = legBull
            cRng = legHigh - legLow
            cNear = (legHigh - near_ratio * cRng) if dirBull else (legLow + near_ratio * cRng)
            w = right + 1
            lo_idx = max(0, i - w + 1)
            if touch_wick:
                win_low = np.min(low[lo_idx : i + 1])
                win_high = np.max(high[lo_idx : i + 1])
                bullLate = win_low <= cNear
                bearLate = win_high >= cNear
            else:
                win_close_lo = np.min(close[lo_idx : i + 1])
                win_close_hi = np.max(close[lo_idx : i + 1])
                bullLate = win_close_lo <= cNear
                bearLate = win_close_hi >= cNear
            lateZone = bullLate if dirBull else bearLate
            zoneFresh = (not skip_late) or (not lateZone)

            if zoneFresh:
                aBull = dirBull
                aSet = True
                aAlive = True
                aRejected = False
                aHigh, aLow = legHigh, legLow
                aBornBar = i
            else:
                if aSet and aAlive:
                    aAlive = False
                    aSet = False

        evBullRej = False
        if aSet and aAlive and aHigh is not None:
            rngA = aHigh - aLow
            if rngA > 0:
                gA = (aHigh - golden_lower * rngA) if aBull else (aLow + golden_lower * rngA)
                gB = (aHigh - golden_upper * rngA) if aBull else (aLow + golden_upper * rngA)
                gTop, gBot = max(gA, gB), min(gA, gB)
                zone_top_arr[i] = gTop
                zone_bot_arr[i] = gBot
                zone_bull_arr[i] = 1.0 if aBull else 0.0

                prevClose = close[i - 1] if i > 0 else np.nan
                prevInside = (not np.isnan(prevClose)) and (prevClose <= gTop) and (prevClose >= gBot)
                touched = (low[i] <= gTop) if touch_wick else prevInside
                bullRejectRaw = aBull and touched and (close[i] > gTop) and (close[i] > openp[i])

                sigEligible = (aBornBar is not None) and (i > aBornBar)
                if sigEligible and bullRejectRaw and not aRejected:
                    aRejected = True
                    evBullRej = True

        longEnterSig = evBullRej or isZigZagLow

        if not position:
            if longEnterSig:
                long_entry[i] = True
                entry_from_gz[i] = evBullRej
                entry_from_zz[i] = (not evBullRej) and isZigZagLow
                position = True
        else:
            if not np.isnan(trailing_stop) and close[i] < trailing_stop:
                long_exit[i] = True
                position = False

    return {
        "long_entry": long_entry,
        "long_exit": long_exit,
        "entry_from_gz": entry_from_gz,
        "entry_from_zz": entry_from_zz,
        "zone_top": zone_top_arr,
        "zone_bot": zone_bot_arr,
        "zone_bull": zone_bull_arr,
        "final_position": position,
        "final_zone": {"bull": aBull, "high": aHigh, "low": aLow, "set": aSet, "alive": aAlive},
        "final_trailing_stop": trailing_stop,
        "atr": atr,
    }


# =============================================================================
# 5. GRAFİK MOTORU
# =============================================================================
def build_chart(df: pd.DataFrame, res: dict, symbol: str, tf_label: str, show_bars: int = 250):
    n = len(df)
    show_n = min(show_bars, n)
    d = df.iloc[-show_n:]
    idx0 = n - show_n

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
            name=symbol, increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        )
    )

    entries = np.where(res["long_entry"][idx0:])[0]
    exits = np.where(res["long_exit"][idx0:])[0]
    if len(entries):
        fig.add_trace(
            go.Scatter(
                x=d.index[entries], y=d["low"].values[entries] * 0.99, mode="markers",
                marker=dict(symbol="triangle-up", size=13, color="#22c55e", line=dict(width=1, color="white")),
                name="AL",
            )
        )
    if len(exits):
        fig.add_trace(
            go.Scatter(
                x=d.index[exits], y=d["high"].values[exits] * 1.01, mode="markers",
                marker=dict(symbol="triangle-down", size=13, color="#ef4444", line=dict(width=1, color="white")),
                name="SAT",
            )
        )

    fz = res["final_zone"]
    if fz["set"] and fz["high"] is not None and fz["low"] is not None:
        rng = fz["high"] - fz["low"]
        if rng > 0:
            gA = fz["high"] - 0.5 * rng if fz["bull"] else fz["low"] + 0.5 * rng
            gB = fz["high"] - 0.618 * rng if fz["bull"] else fz["low"] + 0.618 * rng
            top, bot = max(gA, gB), min(gA, gB)
            fig.add_hrect(
                y0=bot, y1=top, fillcolor="rgba(217,119,6,0.18)", line_width=1,
                line_color="rgba(217,119,6,0.6)", annotation_text="GOLDEN ZONE",
                annotation_position="top left",
            )

    ts = res.get("final_trailing_stop", np.nan)
    if not np.isnan(ts):
        fig.add_hline(y=ts, line_dash="dot", line_color="#facc15", annotation_text="Trailing Stop")

    fig.update_layout(
        template="plotly_dark", height=620, xaxis_rangeslider_visible=False,
        title=f"{symbol} — {tf_label}", margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# =============================================================================
# 6. SIDEBAR — STRATEJİ PARAMETRELERİ VE TARAMA AYARLARI
# =============================================================================
with st.sidebar:
    st.header("⚙️ Tarama Ayarları")

    market_label = st.selectbox("Piyasa", list(MARKET_CONFIGS.keys()))
    tf_label = st.selectbox("Zaman Dilimi", list(TIMEFRAME_CONFIGS.keys()), index=3)
    symbol_limit = st.slider(
        "Taranacak Hisse Sayısı (Piyasa Değerine Göre)", min_value=20, max_value=500, value=120, step=10,
        help="Yüksek sayılarda tarama süresi uzar; yfinance toplu indirme kullanılır.",
    )
    only_after_sell = st.checkbox(
        "Sadece SAT sonrası gelen taze AL sinyallerini göster", value=True,
        help="İşaretliyken, geçmişinde en az bir çıkışı (SAT) olmayan ilk-al sinyalleri listelenmez.",
    )

    with st.expander("📐 Pine Script Strateji Parametreleri", expanded=False):
        pivot_len = st.number_input("Pivot Strength (left bars)", 2, 60, 15)
        confirm_bars = st.number_input("Confirmation Bars (right)", 1, 30, 5)
        golden_lower = st.number_input("Golden Zone Lower Ratio", 0.0, 1.0, 0.5, step=0.01)
        golden_upper = st.number_input("Golden Zone Upper Ratio", 0.0, 1.0, 0.618, step=0.001, format="%.3f")
        inv_buf_atr = st.number_input("Invalidation Buffer (× ATR)", 0.0, 5.0, 0.3, step=0.1)
        zz_dev_atr = st.number_input("Min Leg Size (× ATR)", 0.0, 10.0, 1.5, step=0.1)
        touch_src = st.selectbox("Zone Touch Source", ["Wick", "Close"], index=0)
        skip_late = st.checkbox("Skip late zones", value=True)

    st.write("---")
    run_scan = st.button("🔍 TARAMAYI BAŞLAT")

mkt_cfg = MARKET_CONFIGS[market_label]
tf_cfg = TIMEFRAME_CONFIGS[tf_label]

for key in ["scan_results", "scan_meta"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "scan_results" else {}

# =============================================================================
# 7. TARAMA ÇALIŞTIRMA
# =============================================================================
if run_scan:
    with st.spinner(f"{market_label} sembol listesi alınıyor..."):
        symbols = get_market_symbols(mkt_cfg["tv_market"], limit=symbol_limit)

    if not symbols:
        st.error("Sembol listesi alınamadı. Ağ bağlantınızı veya TradingView scanner erişimini kontrol edin.")
    else:
        yf_tickers = [f"{s.replace('.', '-')}{mkt_cfg['yf_suffix']}" for s in symbols]

        with st.spinner(f"{len(yf_tickers)} hisse için {tf_label} verisi indiriliyor..."):
            batch = fetch_batch(tuple(yf_tickers), tf_cfg["yf_interval"], tf_cfg["period"])

        results = []
        progress = st.progress(0)
        status = st.empty()
        stored_dfs = {}
        stored_res = {}

        single_ticker = len(yf_tickers) == 1

        for idx, (sym, yf_t) in enumerate(zip(symbols, yf_tickers)):
            progress.progress((idx + 1) / len(symbols))
            status.text(f"İşleniyor: {sym} ({idx+1}/{len(symbols)})")

            df = extract_symbol_df(batch, yf_t, single_ticker)
            if df is None or len(df) < MIN_BARS_REQUIRED:
                continue

            if tf_cfg["resample"]:
                try:
                    df = _resample_ohlcv(df, tf_cfg["resample"])
                except Exception:
                    continue
                if len(df) < MIN_BARS_REQUIRED:
                    continue

            try:
                res = run_strategy(
                    df,
                    left=int(pivot_len),
                    right=int(confirm_bars),
                    golden_lower=float(golden_lower),
                    golden_upper=float(golden_upper),
                    inv_buf_atr=float(inv_buf_atr),
                    zz_dev_atr=float(zz_dev_atr),
                    touch_wick=(touch_src == "Wick"),
                    skip_late=skip_late,
                )
            except Exception:
                continue

            n = len(df)
            last_idx = n - 1
            is_fresh_entry_now = bool(res["long_entry"][last_idx])
            had_prior_exit = bool(np.any(res["long_exit"][: last_idx])) if last_idx > 0 else False

            qualifies = is_fresh_entry_now and (had_prior_exit or not only_after_sell)

            if qualifies:
                sig_type = "🟡 Golden Zone Reddi" if res["entry_from_gz"][last_idx] else "🔵 ZigZag Dip Onayı"
                tv_url = (
                    f"https://www.tradingview.com/chart/?symbol={mkt_cfg['tv_prefix']}{sym}"
                    f"&interval={tf_cfg['tv_interval']}"
                )
                fz = res["final_zone"]
                gLow = gTop = np.nan
                if fz["set"] and fz["high"] is not None:
                    rng = fz["high"] - fz["low"]
                    if rng > 0:
                        gA = fz["high"] - golden_lower * rng if fz["bull"] else fz["low"] + golden_lower * rng
                        gB = fz["high"] - golden_upper * rng if fz["bull"] else fz["low"] + golden_upper * rng
                        gTop, gLow = max(gA, gB), min(gA, gB)

                results.append(
                    {
                        "Hisse": sym,
                        "Sinyal Tipi": sig_type,
                        "Güncel Fiyat": round(float(df["close"].iloc[-1]), 4),
                        "Golden Zone Alt": round(float(gLow), 4) if not np.isnan(gLow) else None,
                        "Golden Zone Üst": round(float(gTop), 4) if not np.isnan(gTop) else None,
                        "Trailing Stop": (
                            round(float(res["final_trailing_stop"]), 4)
                            if not np.isnan(res["final_trailing_stop"])
                            else None
                        ),
                        "Önceki SAT Var mı": "Evet" if had_prior_exit else "Hayır (İlk Sinyal)",
                        "Bağlantı": tv_url,
                    }
                )
                stored_dfs[sym] = df
                stored_res[sym] = res

        progress.empty()
        status.empty()

        st.session_state.scan_results = results
        st.session_state.scan_meta = {
            "dfs": stored_dfs,
            "res": stored_res,
            "tf_label": tf_label,
            "market_label": market_label,
            "scanned_count": len(symbols),
        }

        if results:
            st.success(f"Tarama tamamlandı: {len(symbols)} hisse tarandı, {len(results)} hisse TAM AL NOKTASI'nda.")
        else:
            st.warning(f"Tarama tamamlandı: {len(symbols)} hisse tarandı, kriterlere uyan hisse bulunamadı.")

# =============================================================================
# 8. SONUÇ TABLOSU VE GRAFİK İSTASYONU
# =============================================================================
results = st.session_state.scan_results
meta = st.session_state.scan_meta

if results:
    st.write("---")
    st.subheader(f"🎯 TAM AL NOKTASINDAKİ HİSSELER — {meta.get('market_label','')} / {meta.get('tf_label','')}")

    res_df = pd.DataFrame(results)
    st.dataframe(
        res_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Güncel Fiyat": st.column_config.NumberColumn("Güncel Fiyat", format="%.4f"),
            "Golden Zone Alt": st.column_config.NumberColumn("Golden Zone Alt", format="%.4f"),
            "Golden Zone Üst": st.column_config.NumberColumn("Golden Zone Üst", format="%.4f"),
            "Trailing Stop": st.column_config.NumberColumn("Trailing Stop", format="%.4f"),
            "Bağlantı": st.column_config.LinkColumn("TradingView", display_text="📊 Grafiği Aç"),
        },
    )

    st.write("---")
    st.subheader("🔬 Grafik İnceleme İstasyonu")
    symbols_available = list(meta["dfs"].keys())
    selected = st.selectbox("İncelemek için hisse seçin:", symbols_available)

    if selected:
        col_chart, col_info = st.columns([4, 1])
        df_sel = meta["dfs"][selected]
        res_sel = meta["res"][selected]

        with col_chart:
            st.plotly_chart(build_chart(df_sel, res_sel, selected, meta["tf_label"]), use_container_width=True)

        with col_info:
            row = next(r for r in results if r["Hisse"] == selected)
            st.metric("Güncel Fiyat", row["Güncel Fiyat"])
            if row["Golden Zone Alt"] is not None:
                st.metric("Golden Zone", f"{row['Golden Zone Alt']} — {row['Golden Zone Üst']}")
            if row["Trailing Stop"] is not None:
                st.metric("Trailing Stop", row["Trailing Stop"])
            st.write(row["Sinyal Tipi"])
            st.link_button("📊 TradingView'de Aç", row["Bağlantı"])
else:
    st.info(
        "Sol menüden piyasa, zaman dilimi ve strateji parametrelerini seçip **TARAMAYI BAŞLAT** butonuna basın. "
        "Strateji, Pine Script'teki Fibonacci Golden Zone AL/SAT mantığını birebir uygulayarak, sat işleminden "
        "sonra tam alım noktasında olan hisseleri listeler."
    )

st.write("---")
st.caption(
    "⚠️ Bu araç yatırım tavsiyesi değildir. Sinyaller, ekte paylaşılan Pine Script stratejisinin Python "
    "adaptasyonuna dayanır ve gecikmeli/onaylı bar mantığı kullanır. Veriler Yahoo Finance (yfinance) üzerinden, "
    "sembol listeleri ise TradingView public scanner API üzerinden alınır."
)
