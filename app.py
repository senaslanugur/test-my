import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import requests
import warnings

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
# 2. GLOBAL KONFİGÜRASYONLAR VE PERİYOT (TIMEFRAME) AYARLARI
# =============================================================================
MARKET_CONFIGS = {
    "Türkiye (BIST)": {"tv_market": "turkey", "yf_suffix": ".IS", "tv_prefix": "BIST:"},
    "Amerika (ABD)": {"tv_market": "america", "yf_suffix": "", "tv_prefix": ""}
}

# yfinance için 2h ve 4h periyotları doğrudan stabil değildir.
# Bu yüzden 1h verisi çekilip Pandas ile Sentetik (Resample) Mum oluşturulur.
TIMEFRAME_CONFIGS = {
    "1 Saatlik (1H)": {"interval": "1h", "period": "6mo", "resample_rule": None},
    "2 Saatlik (2H)": {"interval": "1h", "period": "6mo", "resample_rule": "2h"},
    "4 Saatlik (4H)": {"interval": "1h", "period": "6mo", "resample_rule": "4h"},
    "1 Günlük (1D)": {"interval": "1d", "period": "1y", "resample_rule": None},
    "1 Haftalık (1W)": {"interval": "1wk", "period": "3y", "resample_rule": None}
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
# 4. KANTİTATİF ALGORİTMA: HİBRİT PINE SCRIPT MİMARİSİ
# =============================================================================
def evaluate_golden_zone(df, signal_window=5):
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    
    # Pine Script ATR Hesaplaması (Gürültü Filtresi İçin)
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
    tr = np.maximum(tr, np.abs(lows[1:] - closes[:-1]))
    tr = np.insert(tr, 0, highs[0] - lows[0])
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1/14) * tr[i] + (1 - 1/14) * atr[i-1]

    pivots = []
    # Pivot Tespiti (15 bar geriye, 5 bar ileriye)
    for i in range(15, len(df) - 5):
        if highs[i] == np.max(highs[i-15 : i+6]):
            pivots.append({'idx': i, 'type': 'H', 'val': highs[i], 'conf_idx': i + 5})
        if lows[i] == np.min(lows[i-15 : i+6]):
            pivots.append({'idx': i, 'type': 'L', 'val': lows[i], 'conf_idx': i + 5})
            
    if len(pivots) < 2: 
        return False, None
        
    # ZigZag Filtrelemesi
    zz = [pivots[0]]
    for p in pivots[1:]:
        last_p = zz[-1]
        if p['type'] == last_p['type']:
            if p['type'] == 'H' and p['val'] > last_p['val']: zz[-1] = p
            elif p['type'] == 'L' and p['val'] < last_p['val']: zz[-1] = p
        else:
            zz.append(p)
            
    if len(zz) < 2: 
        return False, None
        
    last_pivot = zz[-1]
    prev_pivot = zz[-2]
    
    # Golden Zone Kurulumu (Son bacak yükseliş olmalı veya son oluşan pivot dip olmalı)
    if last_pivot['type'] == 'H':
        leg_high = last_pivot['val']
        leg_low = prev_pivot['val']
    else:
        leg_high = prev_pivot['val']
        leg_low = last_pivot['val']
        
    if leg_high <= leg_low: 
        return False, None
        
    rng = leg_high - leg_low
    gz_upper = leg_high - (0.5 * rng)
    gz_lower = leg_high - (0.618 * rng)
    
    curr_close = closes[-1]
    curr_low = lows[-1]
    curr_open = opens[-1]
    
    signal = False
    signal_type = ""
    dist_pct = (curr_close - gz_upper) / gz_upper
    
    # SİNYAL 1: PINE SCRIPT "AL (TREND/DİP)" TEYİT YAKALAYICI
    current_bar_idx = len(df) - 1
    
    if last_pivot['type'] == 'L':
        # Dibin onaylanma barı (conf_idx), son 'signal_window' içine düşüyorsa
        if (current_bar_idx - signal_window) <= last_pivot['conf_idx'] <= current_bar_idx:
            # ATR Filtresi
            min_leg_required = 1.5 * atr[last_pivot['conf_idx']]
            bounce_distance = closes[last_pivot['conf_idx']] - last_pivot['val']
            
            if bounce_distance >= min_leg_required:
                signal = True
                signal_type = "🔥 Al (Trend/Dip) Onayı Geldi!"
    
    # SİNYAL 2: Kusursuz Ret
    if not signal and (curr_low <= gz_upper and curr_close > gz_upper and curr_close > curr_open):
        signal = True
        signal_type = "🎯 Kusursuz Ret (Tam Alım)"
        
    # SİNYAL 3: Pusu Modu
    elif not signal and (gz_lower <= curr_close <= gz_upper):
        signal = True
        signal_type = "⏳ Pusu Modu (Bölge İçi Bekleyiş)"
        
    # SİNYAL 4: Yaklaşanlar
    elif not signal and (0 < dist_pct <= 0.025):
        signal = True
        signal_type = f"👀 Yaklaşıyor (+%{dist_pct*100:.1f})"
        
    if signal:
        return True, {
            "type": signal_type, 
            "gz_lower": gz_lower, 
            "gz_upper": gz_upper,
            "tp": leg_high + (rng * 0.618), 
            "last_high": leg_high, 
            "stop": leg_low * 0.99
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
    width = 0.6
    
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
        
        live_logs.append(f"[SYSTEM]: İndirme tamamlandı. {selected_tf} periyodu için analiz başlıyor...")
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
                
            # SENTETİK MUM (RESAMPLING) MİMARİSİ
            if tf_config["resample_rule"]:
                try:
                    df_symbol.index = pd.to_datetime(df_symbol.index)
                    df_symbol = df_symbol.resample(tf_config["resample_rule"]).agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                except Exception:
                    continue
                
            if len(df_symbol) < 50:
                continue
                
            is_setup, ctx = evaluate_golden_zone(df_symbol)
            
            if is_setup:
                live_logs.append(f"[BULL] {symbol:<6} : {ctx['type']}")
                console_placeholder.code("\n".join(live_logs[-15:]))
                
                curr_price = float(df_symbol['Close'].iloc[-1])
                tv_interval_map = {"1h": "60", "2h": "120", "4h": "240", "1d": "D", "1wk": "W"}
                tv_interval = tv_interval_map.get(tf_config["resample_rule"] or tf_config["interval"], "D")
                tv_url = f"https://www.tradingview.com/chart/?symbol={mkt_config['tv_prefix']}{symbol}&interval={tv_interval}"
                
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
        st.markdown(f"<div style='color:#10b981; font-family:Inter; font-weight:600;'>[SİSTEM BİLGİSİ] {selected_tf} Taraması başarıyla tamamlandı.</div>", unsafe_allow_html=True)
        
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
            st.warning(f"Bu periyotta ({selected_tf}) Altın Bölge kriterlerini karşılayan veya yaklaşan hisse bulunamadı.")
