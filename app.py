import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import requests
import warnings
import time

# Yfinance uyarılarını terminalde gizle
warnings.filterwarnings("ignore")

# =============================================================================
# 1. SAYFA VE PROFESYONEL UX KONFİGÜRASYONU
# =============================================================================
st.set_page_config(page_title="Golden Zone Workstation", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .reportview-container { background: #0a0e17; }
        h1, h2, h3 { font-weight: 800; letter-spacing: -0.5px; color: #e5e7eb; }
        code, .stCodeBlock code { font-family: 'JetBrains Mono', monospace; font-size: 0.85em; color: #10b981; }
        div[data-testid="stExpander"] { background-color: #111827; border: 1px solid #1f2937; border-radius: 2px;}
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. GLOBAL KONFİGÜRASYONLAR
# =============================================================================
MARKET_CONFIGS = {
    "Türkiye (BIST)": {"tv_market": "turkey", "yf_suffix": ".IS", "tv_prefix": "BIST:"},
    "Amerika (ABD)": {"tv_market": "america", "yf_suffix": "", "tv_prefix": ""}
}

TIMEFRAME_CONFIGS = {
    "1 Saatlik (1H)": {"interval": "1h", "period": "1mo"},
    "1 Günlük (1D)": {"interval": "1d", "period": "1y"},
    "1 Haftalık (1W)": {"interval": "1wk", "period": "2y"}
}

# =============================================================================
# 3. VERİ ÇEKME MOTORU (ÇÖKMEZ MİMARİ)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_all_market_symbols(mkt_config):
    url = f"https://scanner.tradingview.com/{mkt_config['tv_market']}/scan"
    payload = {
        "filter": [{"left": "type", "operation": "in_range", "right": ["stock"]}],
        "options": {"lang": "en"}, "markets": [mkt_config['tv_market']],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "volume"],
        "sort": {"sortBy": "volume", "sortOrder": "desc"}, 
        "range": [0, 400] 
    }
    try:
        resp = requests.post(url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code == 200:
            return [item["d"][0] for item in resp.json().get("data", [])]
    except Exception:
        pass
    return []

@st.cache_data(ttl=900, show_spinner=False) 
def fetch_data_cached(tickers, period, interval):
    """Veriyi MultiIndex formatında toplu olarak güvenle indirir."""
    return yf.download(tickers=tickers, period=period, interval=interval, group_by="ticker", threads=True, progress=False)

# =============================================================================
# 4. KANTİTATİF ALGORİTMA: GOLDEN ZONE VEKTÖR MOTORU
# =============================================================================
def evaluate_golden_zone(df):
    highs, lows, closes, opens = df['High'].values, df['Low'].values, df['Close'].values, df['Open'].values
    
    pivots = []
    for i in range(15, len(df) - 5):
        if highs[i] == np.max(highs[i-15 : i+6]): pivots.append((i, 'H', highs[i]))
        if lows[i] == np.min(lows[i-15 : i+6]): pivots.append((i, 'L', lows[i]))
            
    if len(pivots) < 2: return False, None
        
    zz = [pivots[0]]
    for p in pivots[1:]:
        last_p = zz[-1]
        if p[1] == last_p[1]:
            if p[1] == 'H' and p[2] > last_p[2]: zz[-1] = p
            elif p[1] == 'L' and p[2] < last_p[2]: zz[-1] = p
        else: zz.append(p)
            
    if len(zz) < 2: return False, None
        
    last_pivot, prev_pivot = zz[-1], zz[-2]
    if last_pivot[1] != 'H': return False, None
        
    leg_high, leg_low = last_pivot[2], prev_pivot[2]
    if leg_high <= leg_low: return False, None
        
    rng = leg_high - leg_low
    gz_upper = leg_high - (0.5 * rng)
    gz_lower = leg_high - (0.618 * rng)
    
    pullback_bars = df.iloc[last_pivot[0] + 1 : ]
    if pullback_bars.empty: return False, None
        
    touched_zone = pullback_bars['Low'].min() <= gz_upper
    curr_close, curr_low, curr_open = closes[-1], lows[-1], opens[-1]
    
    signal = False
    signal_type = ""
    dist_pct = (curr_close - gz_upper) / gz_upper
    
    if curr_low <= gz_upper and curr_close > gz_upper and curr_close > curr_open:
        signal, signal_type = True, "🎯 Kusursuz Ret (Tam Alım)"
    elif gz_lower <= curr_close <= gz_upper:
        signal, signal_type = True, "⏳ Pusu Modu (Bölge İçi)"
    elif touched_zone and curr_close > gz_upper and curr_close > curr_open:
        signal, signal_type = True, "🚀 Bölgeden Onaylı Çıkış"
    elif 0 < dist_pct <= 0.025:
        signal, signal_type = True, f"👀 Yaklaşıyor (+%{dist_pct*100:.1f})"
        
    if signal:
        return True, {
            "type": signal_type, "gz_lower": gz_lower, "gz_upper": gz_upper,
            "tp": leg_high + (rng * 0.618), "last_high": leg_high, "stop": leg_low * 0.99
        }
    return False, None

# =============================================================================
# 5. PROFESYONEL GRAFİK ÇİZİMİ
# =============================================================================
def draw_gz_chart(symbol, df, ctx):
    plot_df = df.tail(100).copy()
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    up = plot_df['Close'] >= plot_df['Open']
    down = plot_df['Close'] < plot_df['Open']
    width, width2 = 0.6, 0.1
    
    # Profesyonel Mum Grafiği
    ax.bar(plot_df.index[up], plot_df['Close'][up] - plot_df['Open'][up], width, bottom=plot_df['Open'][up], color='#10b981')
    ax.bar(plot_df.index[down], plot_df['Open'][down] - plot_df['Close'][down], width, bottom=plot_df['Close'][down], color='#ef4444')
    ax.vlines(plot_df.index[up], plot_df['Low'][up], plot_df['High'][up], color='#10b981', linewidth=1)
    ax.vlines(plot_df.index[down], plot_df['Low'][down], plot_df['High'][down], color='#ef4444', linewidth=1)
    
    # Golden Zone Çizgileri
    ax.axhspan(ctx["gz_lower"], ctx["gz_upper"], color='#f59e0b', alpha=0.2, label='Altın Bölge (Golden Pocket)')
    ax.axhline(ctx["tp"], color='#00ffff', linestyle='--', linewidth=1.5, label='Kâr Al (1.618 Projeksiyon)')
    ax.axhline(ctx["stop"], color='#ef4444', linestyle=':', linewidth=2, label='Stop Loss (Ana Dip)')
    ax.axhline(ctx["last_high"], color='#9ca3af', linestyle='-', linewidth=1, alpha=0.5, label='Son Zirve')
    
    ax.set_title(f"{symbol} | DURUM: {ctx['type']}", color='#e5e7eb', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper left', frameon=False, labelcolor='white')
    ax.grid(True, alpha=0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig

# =============================================================================
# 6. ARAYÜZ (UI) MİMARİSİ
# =============================================================================
st.title("📈 GOLDEN ZONE WORKSTATION")
st.write("---")

col_mkt, col_tf, col_btn = st.columns([2, 2, 1])
with col_mkt: selected_mkt = st.selectbox("Piyasa Seçin:", list(MARKET_CONFIGS.keys()))
with col_tf: selected_tf = st.selectbox("Zaman Periyodu:", list(TIMEFRAME_CONFIGS.keys()))
with col_btn: 
    st.write("##")
    execute_scan = st.button("TARAMAYI BAŞLAT")

if execute_scan:
    mkt_config = MARKET_CONFIGS[selected_mkt]
    tf_config = TIMEFRAME_CONFIGS[selected_tf]
    
    st.write("### 🖥️ SİSTEM LOG KONSOLU")
    console_placeholder = st.empty()
    live_logs = ["[SYSTEM]: TradingView'den piyasa hacim listesi alınıyor..."]
    console_placeholder.code("\n".join(live_logs))
    
    market_symbols = get_all_market_symbols(mkt_config)
    
    if not market_symbols:
        live_logs.append("[ERROR]: TradingView sunucularına ulaşılamadı.")
        console_placeholder.code("\n".join(live_logs[-15:]))
    else:
        yf_tickers = [f"{s.replace('.', '-')}{mkt_config['yf_suffix']}" for s in market_symbols]
        live_logs.append(f"[SYSTEM]: {len(market_symbols)} hissenin verisi tek pakette indiriliyor...")
        console_placeholder.code("\n".join(live_logs[-15:]))
        
        # Güvenli MultiIndex İndirme
        df_all = fetch_data_cached(yf_tickers, tf_config["period"], tf_config["interval"])
        
        live_logs.append("[SYSTEM]: İndirme tamamlandı. Kuantitatif analiz başlıyor...")
        console_placeholder.code("\n".join(live_logs[-15:]))
        
        p_bar = st.progress(0)
        found_signals = []
        stored_dfs = {}
        
        for idx, symbol in enumerate(market_symbols):
            p_bar.progress((idx + 1) / len(market_symbols))
            yf_ticker_key = f"{symbol.replace('.', '-')}{mkt_config['yf_suffix']}"
            
            # Hata Engelleyici Mimari (hasattr kontrolü)
            if hasattr(df_all.columns, 'levels') and yf_ticker_key in df_all.columns.levels[0]:
                df_symbol = df_all[yf_ticker_key].dropna(subset=['High', 'Close', 'Low', 'Open']).copy()
            elif len(yf_tickers) == 1:
                df_symbol = df_all.dropna(subset=['High', 'Close', 'Low', 'Open']).copy()
            else:
                live_logs.append(f"[SKIP] {symbol:<6} : Veri seti eksik veya tahta kapalı.")
                console_placeholder.code("\n".join(live_logs[-15:]))
                continue
                
            if len(df_symbol) < 50:
                continue
                
            is_setup, ctx = evaluate_golden_zone(df_symbol)
            
            if is_setup:
                live_logs.append(f"[BULL] {symbol:<6} : {ctx['type']}")
                console_placeholder.code("\n".join(live_logs[-15:]))
                
                curr_price = float(df_symbol['Close'].iloc[-1])
                tv_url = f"https://www.tradingview.com/chart/?symbol={mkt_config['tv_prefix']}{symbol}"
                
                found_signals.append({
                    "Durum": ctx["type"],
                    "Hisse": symbol,
                    "Fiyat": round(curr_price, 2),
                    "Giriş (Altın Bölge)": f"{ctx['gz_lower']:.2f} - {ctx['gz_upper']:.2f}",
                    "Kâr Hedefi": round(ctx["tp"], 2),
                    "Stop Loss": round(ctx["stop"], 2),
                    "Bağlantı": tv_url
                })
                stored_dfs[symbol] = {"df": df_symbol, "ctx": ctx}
        
        p_bar.empty()
        st.markdown("<div style='color:#10b981; font-family:Inter; font-weight:600;'>[SİSTEM BİLGİSİ] Tarama işlemi başarıyla tamamlandı.</div>", unsafe_allow_html=True)
        
        if found_signals:
            st.write("---")
            st.write(f"### 🏆 ONAYLANMIŞ FIRSATLAR ({selected_tf})")
            
            res_df = pd.DataFrame(found_signals).sort_values(by="Durum")
            st.dataframe(res_df, use_container_width=True, hide_index=True,
                         column_config={"Bağlantı": st.column_config.LinkColumn("TradingView", display_text="Grafiği Aç")})
            
            st.write("---")
            st.write("### 🔬 GRAFİK İNCELEME İSTASYONU")
            selected_plot = st.selectbox("Detaylı inceleme için hisse seçin:", list(stored_dfs.keys()))
            if selected_plot:
                st.pyplot(draw_gz_chart(selected_plot, stored_dfs[selected_plot]["df"], stored_dfs[selected_plot]["ctx"]))
        else:
            st.warning("Bu periyotta Altın Bölge kriterlerini karşılayan veya yaklaşan hisse bulunamadı.")
