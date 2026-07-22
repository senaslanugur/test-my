import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.subplots as plt_subplots
import matplotlib.dates as mdates
import seaborn as sns
import requests
import warnings

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

TIMEFRAME_CONFIGS = {
    "1 Saatlik (1H)": {"interval": "1h", "period": "6mo", "resample_rule": None},
    "2 Saatlik (2H)": {"interval": "1h", "period": "6mo", "resample_rule": "2h"},
    "4 Saatlik (4H)": {"interval": "1h", "period": "6mo", "resample_rule": "4h"},
    "1 Günlük (1D)": {"interval": "1d", "period": "1y", "resample_rule": None},
    "1 Haftalık (1W)": {"interval": "1wk", "period": "3y", "resample_rule": None}
}

# =============================================================================
# 3. VERİ ÇEKME MOTORU
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
    return yf.download(tickers=tickers, period=period, interval=interval, group_by="ticker", threads=True, progress=False)

# =============================================================================
# 4. KANTİTATİF ALGORİTMA: STATE MACHINE (DURUM HAFIZALI) PINE MİMARİSİ
# =============================================================================
def evaluate_golden_zone_state_machine(df, signal_window=5):
    pivotLen = 15
    confirmBars = 5
    zzDevAtr = 1.5
    invBufAtr = 0.3
    goldenLower = 0.5
    goldenUpper = 0.618
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    
    # ATR (RMA) Hesaplaması
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
    tr = np.maximum(tr, np.abs(lows[1:] - closes[:-1]))
    tr = np.insert(tr, 0, highs[0] - lows[0])
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 1.0 / 14
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
    N = len(highs)
    if N < pivotLen + confirmBars:
        return False, None
        
    zzP1 = np.nan; zzP0 = np.nan
    zzD1 = 0
    zzLow = np.nan; zzHigh = np.nan
    
    aSet = False; aAlive = False
    aHigh = np.nan; aLow = np.nan
    gTop = np.nan; gBot = np.nan
    aRejected = False
    
    trailing_stop = np.nan
    in_position = False  # Portföy Durum Hafızası (Sanal İşlem)
    
    signal_triggered = False
    signal_data = {}
    
    for i in range(pivotLen + confirmBars, N):
        isZigZagHigh = False
        isZigZagLow = False
        zzLegEvent = False
        
        idx_eval = i - confirmBars
        window_high = highs[idx_eval - pivotLen : i + 1]
        window_low = lows[idx_eval - pivotLen : i + 1]
        
        usePH = np.nan; usePL = np.nan
        if highs[idx_eval] == np.max(window_high): usePH = highs[idx_eval]
        if lows[idx_eval] == np.min(window_low): usePL = lows[idx_eval]
            
        if not np.isnan(usePH) and not np.isnan(usePL):
            if np.isnan(zzP1):
                usePH = np.nan; usePL = np.nan
            else:
                dH = abs(usePH - zzP1); dL = abs(usePL - zzP1)
                if dH > dL: usePL = np.nan
                elif dL > dH: usePH = np.nan
                else: usePH = np.nan; usePL = np.nan
        
        pivotAtr = atr[idx_eval] if not np.isnan(atr[idx_eval]) else 0
        zzMinLeg = zzDevAtr * pivotAtr
        
        if not np.isnan(usePH):
            if zzD1 == 1:
                if usePH > zzP1: zzP1 = usePH; zzHigh = usePH; zzLegEvent = True; isZigZagHigh = True
            elif np.isnan(zzP1): zzP1 = usePH; zzD1 = 1; zzHigh = usePH; isZigZagHigh = True
            elif abs(usePH - zzP1) > zzMinLeg: zzP0 = zzP1; zzP1 = usePH; zzD1 = 1; zzHigh = usePH; zzLegEvent = True; isZigZagHigh = True
                
        if not np.isnan(usePL):
            if zzD1 == -1:
                if usePL < zzP1: zzP1 = usePL; zzLow = usePL; zzLegEvent = True; isZigZagLow = True
            elif np.isnan(zzP1): zzP1 = usePL; zzD1 = -1; zzLow = usePL; isZigZagLow = True
            elif abs(usePL - zzP1) > zzMinLeg: zzP0 = zzP1; zzP1 = usePL; zzD1 = -1; zzLow = usePL; zzLegEvent = True; isZigZagLow = True
                
        # ATR Trailing Stop Güncellemesi (Sadece dip oluştuğunda)
        if isZigZagLow and not np.isnan(zzLow):
            trailing_stop = zzLow - (invBufAtr * atr[i])
            
        validLeg = (zzD1 != 0) and not np.isnan(zzP0) and not np.isnan(zzP1) and (zzP0 != zzP1)
        dirBull = (zzD1 == 1)
        legHigh = max(zzP0, zzP1) if validLeg else np.nan
        legLow = min(zzP0, zzP1) if validLeg else np.nan
        
        validSetup = validLeg and (legHigh > legLow)
        
        if zzLegEvent and validSetup:
            aSet = True; aAlive = True
            aHigh = legHigh; aLow = legLow
            aRejected = False
            rng = aHigh - aLow
            gTop = aHigh - (goldenLower * rng)
            gBot = aHigh - (goldenUpper * rng)
            
        activeValid = aSet and aAlive and not np.isnan(aHigh) and not np.isnan(aLow) and (aHigh - aLow) > 0
        evBullRej = False
        
        if activeValid and dirBull:
            touchWick = (lows[i] <= gTop)
            bullRejectRaw = touchWick and (closes[i] > gTop) and (closes[i] > opens[i])
            if bullRejectRaw and not aRejected:
                aRejected = True
                evBullRej = True
                
        longEnter = (evBullRej or isZigZagLow) and activeValid and dirBull
        longExit = (not np.isnan(trailing_stop)) and (closes[i] < trailing_stop)
        
        # --- DURUM MAKİNESİ (Sanal Portföy & Pine Script İşlem Yönetimi) ---
        current_bar_signal = None
        
        if in_position:
            if longExit:
                in_position = False
                trailing_stop = np.nan # Stop patladı, satıldı
                current_bar_signal = "Sat (Trend Bozuldu)"
            elif longEnter:
                current_bar_signal = "📈 Trend İçi Ekleme (Zaten Al'da)"
        else:
            if longEnter:
                in_position = True
                if evBullRej:
                    current_bar_signal = "🎯 Taze Alım (Kusursuz Ret)"
                elif isZigZagLow:
                    current_bar_signal = "🔥 Taze Alım (Trend/Dip Onayı)"
        
        # Sinyal Penceresi Taraması
        if i >= N - signal_window:
            if current_bar_signal and "Sat" not in current_bar_signal:
                signal_triggered = True
                bars_ago = (N - 1) - i
                ext = f" ({bars_ago} bar önce)" if bars_ago > 0 else ""
                
                signal_data = {
                    "type": current_bar_signal + ext,
                    "gz_lower": gBot, "gz_upper": gTop,
                    "tp": aHigh + ((aHigh - aLow) * 0.618),
                    "last_high": aHigh,
                    "stop": trailing_stop if not np.isnan(trailing_stop) else aLow
                }
                
    # O ANKİ bar için Pusu veya Yaklaşma durumu (Sadece Nakitteysek geçerli)
    if not signal_triggered and activeValid and dirBull and not in_position:
        curr_close = closes[-1]
        dist_pct = (curr_close - gTop) / gTop
        if gBot <= curr_close <= gTop:
            signal_triggered = True
            signal_data = {
                "type": "⏳ Pusu Modu (Nakitte, Bölge İçi Bekleyiş)",
                "gz_lower": gBot, "gz_upper": gTop,
                "tp": aHigh + ((aHigh - aLow) * 0.618),
                "last_high": aHigh,
                "stop": trailing_stop if not np.isnan(trailing_stop) else aLow
            }
        elif 0 < dist_pct <= 0.025:
            signal_triggered = True
            signal_data = {
                "type": f"👀 Yaklaşıyor (+%{dist_pct*100:.1f}) (Nakitte)",
                "gz_lower": gBot, "gz_upper": gTop,
                "tp": aHigh + ((aHigh - aLow) * 0.618),
                "last_high": aHigh,
                "stop": trailing_stop if not np.isnan(trailing_stop) else aLow
            }

    if signal_triggered:
        return True, signal_data
        
    return False, None

# =============================================================================
# 5. PROFESYONEL GRAFİK ÇİZİMİ
# =============================================================================
def draw_gz_chart(symbol, df, ctx):
    plot_df = df.tail(100).copy()
    plt_subplots.style.use('dark_background')
    fig, ax = plt_subplots.subplots(figsize=(12, 6))
    
    up = plot_df['Close'] >= plot_df['Open']
    down = plot_df['Close'] < plot_df['Open']
    width = 0.6
    
    ax.bar(plot_df.index[up], plot_df['Close'][up] - plot_df['Open'][up], width, bottom=plot_df['Open'][up], color='#10b981')
    ax.bar(plot_df.index[down], plot_df['Open'][down] - plot_df['Close'][down], width, bottom=plot_df['Close'][down], color='#ef4444')
    ax.vlines(plot_df.index[up], plot_df['Low'][up], plot_df['High'][up], color='#10b981', linewidth=1)
    ax.vlines(plot_df.index[down], plot_df['Low'][down], plot_df['High'][down], color='#ef4444', linewidth=1)
    
    ax.axhspan(ctx["gz_lower"], ctx["gz_upper"], color='#f59e0b', alpha=0.2, label='Altın Bölge (Golden Pocket)')
    ax.axhline(ctx["tp"], color='#00ffff', linestyle='--', linewidth=1.5, label='Kâr Al (1.618 Projeksiyon)')
    
    if not np.isnan(ctx["stop"]):
        ax.axhline(ctx["stop"], color='#ef4444', linestyle=':', linewidth=2.5, label='ATR İzleyen Stop Loss')
        
    ax.axhline(ctx["last_high"], color='#9ca3af', linestyle='-', linewidth=1, alpha=0.5, label='Son Zirve')
    
    ax.set_title(f"{symbol} | DURUM: {ctx['type']}", color='#e5e7eb', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper left', frameon=False, labelcolor='white')
    ax.grid(True, alpha=0.1)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt_subplots.tight_layout()
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
        
        df_all = fetch_data_cached(yf_tickers, tf_config["period"], tf_config["interval"])
        
        live_logs.append(f"[SYSTEM]: İndirme tamamlandı. {selected_tf} periyodu için durum makinesi analiz başlıyor...")
        console_placeholder.code("\n".join(live_logs[-15:]))
        
        p_bar = st.progress(0)
        found_signals = []
        stored_dfs = {}
        
        for idx, symbol in enumerate(market_symbols):
            p_bar.progress((idx + 1) / len(market_symbols))
            yf_ticker_key = f"{symbol.replace('.', '-')}{mkt_config['yf_suffix']}"
            
            if hasattr(df_all.columns, 'levels') and yf_ticker_key in df_all.columns.levels[0]:
                df_symbol = df_all[yf_ticker_key].dropna(subset=['High', 'Close', 'Low', 'Open']).copy()
            elif len(yf_tickers) == 1:
                df_symbol = df_all.dropna(subset=['High', 'Close', 'Low', 'Open']).copy()
            else:
                live_logs.append(f"[SKIP] {symbol:<6} : Veri seti eksik veya tahta kapalı.")
                console_placeholder.code("\n".join(live_logs[-15:]))
                continue
                
            if tf_config["resample_rule"]:
                try:
                    df_symbol.index = pd.to_datetime(df_symbol.index)
                    df_symbol = df_symbol.resample(tf_config["resample_rule"]).agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                except Exception:
                    continue
                
            if len(df_symbol) < 50:
                continue
                
            is_setup, ctx = evaluate_golden_zone_state_machine(df_symbol)
            
            if is_setup:
                # Loglara sade halini bas
                live_logs.append(f"[MATCH] {symbol:<6} : {ctx['type'].split('(')[0].strip()}")
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
            st.write(f"### 🏆 ONAYLANMIŞ FIRSATLAR VE DURUMLAR ({selected_tf})")
            
            # Taze alımları en üste alacak şekilde basit sıralama
            res_df = pd.DataFrame(found_signals).sort_values(by="Durum", ascending=False)
            st.dataframe(res_df, use_container_width=True, hide_index=True,
                         column_config={"Bağlantı": st.column_config.LinkColumn("TradingView", display_text="Grafiği Aç")})
            
            st.write("---")
            st.write("### 🔬 GRAFİK İNCELEME İSTASYONU")
            selected_plot = st.selectbox("Detaylı inceleme için hisse seçin:", list(stored_dfs.keys()))
            if selected_plot:
                st.pyplot(draw_gz_chart(selected_plot, stored_dfs[selected_plot]["df"], stored_dfs[selected_plot]["ctx"]))
        else:
            st.warning(f"Bu periyotta ({selected_tf}) tespit edilen aktif bir durum bulunamadı.")
