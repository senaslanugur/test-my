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
    addon_signal = np.zeros(n, dtype=bool)
    addon_from_gz = np.zeros(n, dtype=bool)
    addon_from_zz = np.zeros(n, dtype=bool)
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
            # Zaten pozisyondayken gelen yeni bir AL sinyali: Pine'daki pyramiding=1
            # default'u nedeniyle pozisyon büyümez, ama bu bir "ekleme / ikinci alım"
            # fırsatı olarak kullanıcıya ayrıca raporlanır.
            if longEnterSig:
                addon_signal[i] = True
                addon_from_gz[i] = evBullRej
                addon_from_zz[i] = (not evBullRej) and isZigZagLow
            if not np.isnan(trailing_stop) and close[i] < trailing_stop:
                long_exit[i] = True
                position = False

    open_entry_idx = None
    if position:
        entry_positions = np.where(long_entry)[0]
        if len(entry_positions):
            open_entry_idx = int(entry_positions[-1])

    return {
        "long_entry": long_entry,
        "long_exit": long_exit,
        "entry_from_gz": entry_from_gz,
        "entry_from_zz": entry_from_zz,
        "addon_signal": addon_signal,
        "addon_from_gz": addon_from_gz,
        "addon_from_zz": addon_from_zz,
        "zone_top": zone_top_arr,
        "zone_bot": zone_bot_arr,
        "zone_bull": zone_bull_arr,
        "final_position": position,
        "open_entry_idx": open_entry_idx,
        "final_zone": {
            "bull": aBull, "high": aHigh, "low": aLow,
            "set": aSet, "alive": aAlive, "rejected": aRejected,
        },
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
    recency_bars = st.slider(
        "Sinyal Tazeliği (son kaç bar içinde tetiklendi?)", min_value=1, max_value=20, value=3,
        help="AL / Ekleme sinyalinin 'taze' sayılması için son kaç barda oluşmuş olması gerektiği. "
             "Sadece son bar aranırsa çok az/hiç sonuç çıkabilir; bu pencereyi genişletmek sonuç sayısını artırır.",
    )
    watch_atr_mult = st.slider(
        "Yakın Takip Mesafesi (× ATR)", min_value=0.2, max_value=5.0, value=1.5, step=0.1,
        help="Fiyatın Golden Zone'a bu ATR çarpanı kadar mesafede olması, hisseyi 'Yakın Takip' listesine sokar.",
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

for key, default in [
    ("fresh_results", []), ("watch_results", []), ("addon_results", []), ("scan_meta", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _golden_bounds(fz, g_lower, g_upper):
    if not fz["set"] or fz["high"] is None or fz["low"] is None:
        return np.nan, np.nan
    rng = fz["high"] - fz["low"]
    if rng <= 0:
        return np.nan, np.nan
    gA = fz["high"] - g_lower * rng if fz["bull"] else fz["low"] + g_lower * rng
    gB = fz["high"] - g_upper * rng if fz["bull"] else fz["low"] + g_upper * rng
    return min(gA, gB), max(gA, gB)


def _tv_url(mkt_cfg, tf_cfg, sym):
    return f"https://www.tradingview.com/chart/?symbol={mkt_cfg['tv_prefix']}{sym}&interval={tf_cfg['tv_interval']}"


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

        fresh_results, watch_results, addon_results = [], [], []
        progress = st.progress(0)
        status = st.empty()
        stored_dfs, stored_res = {}, {}
        single_ticker = len(yf_tickers) == 1
        processed_ok = 0

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

            processed_ok += 1
            n = len(df)
            last_idx = n - 1
            window_start = max(0, n - int(recency_bars))
            price_now = float(df["close"].iloc[-1])
            tv_url = _tv_url(mkt_cfg, tf_cfg, sym)
            symbol_used = False

            # --- 1) TAZE AL SİNYALİ (son N barda tetiklenen yeni giriş) ---
            entry_hits = np.where(res["long_entry"][window_start:])[0]
            if len(entry_hits):
                entry_idx = window_start + int(entry_hits[-1])
                had_prior_exit = bool(np.any(res["long_exit"][:entry_idx]))
                if had_prior_exit or not only_after_sell:
                    sig_type = "🟡 Golden Zone Reddi" if res["entry_from_gz"][entry_idx] else "🔵 ZigZag Dip Onayı"
                    gLow, gTop = _golden_bounds(res["final_zone"], golden_lower, golden_upper)
                    fresh_results.append({
                        "Hisse": sym,
                        "Sinyal Tipi": sig_type,
                        "Sinyal Bar (geriye dönük)": n - 1 - entry_idx,
                        "Güncel Fiyat": round(price_now, 4),
                        "Golden Zone Alt": round(float(gLow), 4) if not np.isnan(gLow) else None,
                        "Golden Zone Üst": round(float(gTop), 4) if not np.isnan(gTop) else None,
                        "Trailing Stop": (
                            round(float(res["final_trailing_stop"]), 4)
                            if not np.isnan(res["final_trailing_stop"]) else None
                        ),
                        "Önceki SAT Var mı": "Evet" if had_prior_exit else "Hayır (İlk Sinyal)",
                        "Bağlantı": tv_url,
                    })
                    symbol_used = True

            # --- 2) EKLEME / İKİNCİ ALIM NOKTASI (pozisyon açıkken gelen yeni sinyal) ---
            addon_hits = np.where(res["addon_signal"][window_start:])[0]
            if res["final_position"] and len(addon_hits):
                addon_idx = window_start + int(addon_hits[-1])
                sig_type = "🟡 Golden Zone Reddi" if res["addon_from_gz"][addon_idx] else "🔵 ZigZag Dip Onayı"
                gLow, gTop = _golden_bounds(res["final_zone"], golden_lower, golden_upper)
                open_idx = res.get("open_entry_idx")
                open_price = round(float(df["close"].iloc[open_idx]), 4) if open_idx is not None else None
                addon_results.append({
                    "Hisse": sym,
                    "Ekleme Sinyal Tipi": sig_type,
                    "Sinyal Bar (geriye dönük)": n - 1 - addon_idx,
                    "Güncel Fiyat": round(price_now, 4),
                    "İlk Alım Fiyatı": open_price,
                    "Golden Zone Alt": round(float(gLow), 4) if not np.isnan(gLow) else None,
                    "Golden Zone Üst": round(float(gTop), 4) if not np.isnan(gTop) else None,
                    "Trailing Stop": (
                        round(float(res["final_trailing_stop"]), 4)
                        if not np.isnan(res["final_trailing_stop"]) else None
                    ),
                    "Bağlantı": tv_url,
                })
                symbol_used = True

            # --- 3) YAKIN TAKİP (Golden Zone'a yaklaşan / içinde onay bekleyen, henüz pozisyonsuz) ---
            fz = res["final_zone"]
            if (not res["final_position"]) and fz["set"] and fz["alive"] and fz["bull"] and not fz["rejected"]:
                gLow, gTop = _golden_bounds(fz, golden_lower, golden_upper)
                atr_last = res["atr"][last_idx]
                if not np.isnan(gLow) and not np.isnan(atr_last):
                    last_low = float(df["low"].iloc[-1])
                    invalid_level = gLow - watch_atr_mult * atr_last
                    if price_now >= invalid_level:
                        if last_low <= gTop and price_now >= gLow - 0.15 * atr_last:
                            status_txt = "🟠 Bölgede — Onay Bekleniyor"
                        elif 0 < (price_now - gTop) <= watch_atr_mult * atr_last:
                            status_txt = "👀 Yaklaşıyor"
                        else:
                            status_txt = None
                        if status_txt:
                            watch_results.append({
                                "Hisse": sym,
                                "Durum": status_txt,
                                "Güncel Fiyat": round(price_now, 4),
                                "Golden Zone Alt": round(float(gLow), 4),
                                "Golden Zone Üst": round(float(gTop), 4),
                                "Zone'a Uzaklık (ATR)": round((price_now - gTop) / atr_last, 2) if atr_last > 0 else None,
                                "Bağlantı": tv_url,
                            })
                            symbol_used = True

            if symbol_used:
                stored_dfs[sym] = df
                stored_res[sym] = res

        progress.empty()
        status.empty()

        st.session_state.fresh_results = fresh_results
        st.session_state.watch_results = watch_results
        st.session_state.addon_results = addon_results
        st.session_state.scan_meta = {
            "dfs": stored_dfs, "res": stored_res, "tf_label": tf_label,
            "market_label": market_label, "scanned_count": len(symbols), "processed_ok": processed_ok,
        }

        total_hits = len(fresh_results) + len(watch_results) + len(addon_results)
        if total_hits:
            st.success(
                f"Tarama tamamlandı: {len(symbols)} hisse hedeflendi, {processed_ok} hisse başarıyla işlendi. "
                f"AL: {len(fresh_results)}  |  Yakın Takip: {len(watch_results)}  |  Ekleme: {len(addon_results)}"
            )
        else:
            st.warning(
                f"Tarama tamamlandı: {len(symbols)} hisse hedeflendi, {processed_ok} hisse başarıyla işlendi, "
                "hiçbir kategoriye uyan hisse bulunamadı. 'Sinyal Tazeliği' penceresini genişletmeyi "
                "veya taranacak hisse sayısını artırmayı deneyin."
            )
        if processed_ok == 0:
            st.error(
                "Hiçbir hisse verisi işlenemedi — yfinance/ağ erişiminde bir sorun olabilir "
                "(sembol formatı, veri sağlayıcı kısıtı, ya da seçilen zaman diliminde yeterli geçmiş bulunmaması)."
            )

# =============================================================================
# 8. SONUÇ TABLOLARI VE GRAFİK İSTASYONU
# =============================================================================
fresh_results = st.session_state.fresh_results
watch_results = st.session_state.watch_results
addon_results = st.session_state.addon_results
meta = st.session_state.scan_meta

any_results = fresh_results or watch_results or addon_results

if any_results:
    st.write("---")
    st.subheader(f"{meta.get('market_label','')} / {meta.get('tf_label','')} — Tarama Sonuçları")

    tab1, tab2, tab3 = st.tabs([
        f"🎯 Taze AL Sinyalleri ({len(fresh_results)})",
        f"👀 Yakın Takip ({len(watch_results)})",
        f"➕ Ekleme / İkinci Alım Noktası ({len(addon_results)})",
    ])

    with tab1:
        if fresh_results:
            st.dataframe(
                pd.DataFrame(fresh_results), use_container_width=True, hide_index=True,
                column_config={
                    "Güncel Fiyat": st.column_config.NumberColumn(format="%.4f"),
                    "Golden Zone Alt": st.column_config.NumberColumn(format="%.4f"),
                    "Golden Zone Üst": st.column_config.NumberColumn(format="%.4f"),
                    "Trailing Stop": st.column_config.NumberColumn(format="%.4f"),
                    "Bağlantı": st.column_config.LinkColumn("TradingView", display_text="📊 Grafiği Aç"),
                },
            )
        else:
            st.info("Bu kategoriye uyan hisse bulunamadı.")

    with tab2:
        st.caption("Golden Zone'a yaklaşan veya bölge içinde yeşil-mum onayı bekleyen, henüz pozisyona girilmemiş hisseler.")
        if watch_results:
            st.dataframe(
                pd.DataFrame(watch_results), use_container_width=True, hide_index=True,
                column_config={
                    "Güncel Fiyat": st.column_config.NumberColumn(format="%.4f"),
                    "Golden Zone Alt": st.column_config.NumberColumn(format="%.4f"),
                    "Golden Zone Üst": st.column_config.NumberColumn(format="%.4f"),
                    "Bağlantı": st.column_config.LinkColumn("TradingView", display_text="📊 Grafiği Aç"),
                },
            )
        else:
            st.info("Bu kategoriye uyan hisse bulunamadı.")

    with tab3:
        st.caption("Zaten pozisyonda olup, Pine'ın pyramiding kısıtı nedeniyle otomatik büyümeyen ama yeni bir AL/HL sinyali üretmiş — manuel ekleme fırsatı olabilecek hisseler.")
        if addon_results:
            st.dataframe(
                pd.DataFrame(addon_results), use_container_width=True, hide_index=True,
                column_config={
                    "Güncel Fiyat": st.column_config.NumberColumn(format="%.4f"),
                    "İlk Alım Fiyatı": st.column_config.NumberColumn(format="%.4f"),
                    "Golden Zone Alt": st.column_config.NumberColumn(format="%.4f"),
                    "Golden Zone Üst": st.column_config.NumberColumn(format="%.4f"),
                    "Trailing Stop": st.column_config.NumberColumn(format="%.4f"),
                    "Bağlantı": st.column_config.LinkColumn("TradingView", display_text="📊 Grafiği Aç"),
                },
            )
        else:
            st.info("Bu kategoriye uyan hisse bulunamadı.")

    st.write("---")
    st.subheader("🔬 Grafik İnceleme İstasyonu")
    symbols_available = list(meta["dfs"].keys())
    if symbols_available:
        selected = st.selectbox("İncelemek için hisse seçin:", symbols_available)
        if selected:
            col_chart, col_info = st.columns([4, 1])
            df_sel = meta["dfs"][selected]
            res_sel = meta["res"][selected]

            with col_chart:
                st.plotly_chart(build_chart(df_sel, res_sel, selected, meta["tf_label"]), use_container_width=True)

            with col_info:
                st.metric("Güncel Fiyat", round(float(df_sel["close"].iloc[-1]), 4))
                gLow, gTop = _golden_bounds(res_sel["final_zone"], golden_lower, golden_upper)
                if not np.isnan(gLow):
                    st.metric("Golden Zone", f"{round(float(gLow),4)} — {round(float(gTop),4)}")
                ts = res_sel.get("final_trailing_stop", np.nan)
                if not np.isnan(ts):
                    st.metric("Trailing Stop", round(float(ts), 4))
                st.write("Pozisyon: " + ("🟢 Açık (Long)" if res_sel["final_position"] else "⚪ Flat"))
                tv_link = _tv_url(mkt_cfg, tf_cfg, selected)
                st.link_button("📊 TradingView'de Aç", tv_link)
else:
    st.info(
        "Sol menüden piyasa, zaman dilimi ve strateji parametrelerini seçip **TARAMAYI BAŞLAT** butonuna basın. "
        "Strateji üç ayrı liste üretir: (1) taze AL sinyali verenler, (2) Golden Zone'a yaklaşan/içinde bekleyen "
        "izleme listesi, (3) zaten pozisyondayken yeni bir AL sinyali almış (ekleme/ikinci alım) hisseler."
    )

st.write("---")
st.caption(
    "⚠️ Bu araç yatırım tavsiyesi değildir. Sinyaller, ekte paylaşılan Pine Script stratejisinin Python "
    "adaptasyonuna dayanır ve gecikmeli/onaylı bar mantığı kullanır. Veriler Yahoo Finance (yfinance) üzerinden, "
    "sembol listeleri ise TradingView public scanner API üzerinden alınır."
)
