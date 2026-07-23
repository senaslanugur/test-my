import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt_subplots
import matplotlib.dates as mdates
import seaborn as sns
import requests
import warnings
from datetime import datetime, timedelta

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
        .metric-card { background-color: #1f2937; padding: 15px; border-radius: 5px; border-left: 4px solid #3b82f6; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #e5e7eb; }
        .metric-label { font-size: 12px; color: #9ca3af; text-transform: uppercase; }
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
    "1 Saatlik (1H)": {"interval": "1h", "period": "730d", "resample_rule": None},
    "2 Saatlik (2H)": {"interval": "1h", "period": "730d", "resample_rule": "2h"},
    "4 Saatlik (4H)": {"interval": "1h", "period": "730d", "resample_rule": "4h"},
    "1 Günlük (1D)": {"interval": "1d", "period": "max", "resample_rule": None},
    "1 Haftalık (1W)": {"interval": "1wk", "period": "max", "resample_rule": None}
}

# =============================================================================
# 3. VERİ ÇEKME MOTORU VE OTONOM TARİH BULUCU
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
            return sorted([item["d"][0] for item in resp.json().get("data", [])])
    except Exception:
        pass
    return []

@st.cache_data(ttl=900, show_spinner=False) 
def fetch_data_cached(tickers, period, interval):
    return yf.download(tickers=tickers, period=period, interval=interval, group_by="ticker", threads=True, progress=False)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_single_historical_data(ticker, interval, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

@st.cache_data(ttl=86400, show_spinner=False)
def get_first_available_date(ticker, interval):
    try:
        if interval == "1h": return datetime.today().date() - timedelta(days=725)
        hist = yf.download(ticker, period="max", interval="1d", progress=False)
        if not hist.empty: return hist.index.min().date()
    except Exception: pass
    return datetime.today().date() - timedelta(days=365)

# =============================================================================
# 4. KANTİTATİF ALGORİTMA: STATE MACHINE (PINE SCRIPT BİREBİR)
# =============================================================================
def core_pine_state_machine(df):
    pivotLen, confirmBars, zzDevAtr, invBufAtr = 15, 5, 1.5, 0.3
    goldenLower, goldenUpper = 0.5, 0.618
    
    highs, lows = df['High'].values, df['Low'].values
    closes, opens = df['Close'].values, df['Open'].values
    dates = df.index
    
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
    tr = np.maximum(tr, np.abs(lows[1:] - closes[:-1]))
    tr = np.insert(tr, 0, highs[0] - lows[0])
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)): atr[i] = (1/14) * tr[i] + (1 - 1/14) * atr[i-1]
        
    N = len(highs)
    
    zzP1, zzP0, zzD1, zzLow, zzHigh = np.nan, np.nan, 0, np.nan, np.nan
    aSet, aAlive, aBull, aHigh, aLow, gTop, gBot, aRejected = False, False, False, np.nan, np.nan, np.nan, np.nan, False
    trailing_stop = np.nan
    in_position = False
    
    trades = []
    entry_price, entry_date, entry_type = 0, None, ""
    latest_state = {"signal": False, "data": None}
    
    for i in range(pivotLen + confirmBars, N):
        isZigZagHigh, isZigZagLow, zzLegEvent = False, False, False
        idx_eval = i - confirmBars
        window_high, window_low = highs[idx_eval - pivotLen : i + 1], lows[idx_eval - pivotLen : i + 1]
        
        usePH, usePL = np.nan, np.nan
        if highs[idx_eval] == np.max(window_high): usePH = highs[idx_eval]
        if lows[idx_eval] == np.min(window_low): usePL = lows[idx_eval]
            
        if not np.isnan(usePH) and not np.isnan(usePL):
            if np.isnan(zzP1): usePH, usePL = np.nan, np.nan
            else:
                dH, dL = abs(usePH - zzP1), abs(usePL - zzP1)
                if dH > dL: usePL = np.nan
                elif dL > dH: usePH = np.nan
                else: usePH, usePL = np.nan, np.nan
        
        zzMinLeg = zzDevAtr * (atr[idx_eval] if not np.isnan(atr[idx_eval]) else 0)
        
        if not np.isnan(usePH):
            if zzD1 == 1:
                if usePH > zzP1: zzP1 = zzHigh = usePH; zzLegEvent = isZigZagHigh = True
            elif np.isnan(zzP1): zzP1 = zzHigh = usePH; zzD1 = 1; isZigZagHigh = True
            elif abs(usePH - zzP1) > zzMinLeg: zzP0 = zzP1; zzP1 = zzHigh = usePH; zzD1 = 1; zzLegEvent = isZigZagHigh = True
                
        if not np.isnan(usePL):
            if zzD1 == -1:
                if usePL < zzP1: zzP1 = zzLow = usePL; zzLegEvent = isZigZagLow = True
            elif np.isnan(zzP1): zzP1 = zzLow = usePL; zzD1 = -1; isZigZagLow = True
            elif abs(usePL - zzP1) > zzMinLeg: zzP0 = zzP1; zzP1 = zzLow = usePL; zzD1 = -1; zzLegEvent = isZigZagLow = True
                
        if isZigZagLow and not np.isnan(zzLow): trailing_stop = zzLow - (invBufAtr * atr[i])
            
        validLeg = (zzD1 != 0) and not np.isnan(zzP0) and not np.isnan(zzP1) and (zzP0 != zzP1)
        dirBull = (zzD1 == 1)
        
        legHigh = max(zzP0, zzP1) if validLeg else np.nan
        legLow = min(zzP0, zzP1) if validLeg else np.nan
        
        if zzLegEvent and (validLeg and (legHigh > legLow)):
            aSet, aAlive, aBull, aHigh, aLow, aRejected = True, True, dirBull, legHigh, legLow, False
            rng = aHigh - aLow
            gTop = aHigh - (goldenLower * rng)
            gBot = aHigh - (goldenUpper * rng)
            
        activeValid = aSet and aAlive and not np.isnan(aHigh) and not np.isnan(aLow) and (aHigh - aLow) > 0
        evBullRej = False
        
        if activeValid and aBull:
            if (lows[i] <= gTop) and (closes[i] > gTop) and (closes[i] > opens[i]) and not aRejected:
                aRejected = True; evBullRej = True
                
        # --- PINE SCRIPT BİREBİR GİRİŞ VE ÇIKIŞ (SAT) MANTIĞI ---
        longEnter = (evBullRej or isZigZagLow) and activeValid and dirBull
        
        # DÜZELTME: Çıkış şartı ARTIK SADECE ATR İzleyen Stop'un kırılmasıdır.
        # "isZigZagHigh" (Tepe Onayı) şartı kaldırıldı.
        longExit = (not np.isnan(trailing_stop)) and (closes[i] < trailing_stop)
        
        # --- İŞLEM YÖNETİMİ ---
        if in_position:
            if longExit:
                in_position = False # Stop patladı, Nakite Geç
                exit_price = closes[i]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                trades.append({
                    'Giriş Tarihi': entry_date, 'Giriş Fiyatı': entry_price, 'Tetikleyici': entry_type,
                    'Çıkış Tarihi': dates[i], 'Çıkış Fiyatı': exit_price, 'Kâr/Zarar (%)': round(pnl_pct, 2)
                })
                trailing_stop = np.nan
        else:
            if longEnter:
                in_position = True # Mal Alındı
                entry_price, entry_date = closes[i], dates[i]
                entry_type = "🎯 Kusursuz Ret" if evBullRej else "🔥 Trend/Dip Onayı"
                
        # --- TARAYICI DURUMU ---
        if i >= N - 5: 
            current_bar_signal = None
            if in_position and longEnter and i == N-1: 
                current_bar_signal = "📈 Trend İçi Ekleme"
            elif not in_position and longEnter: 
                current_bar_signal = "🎯 Taze Alım (Kusursuz Ret)" if evBullRej else "🔥 Taze Alım (Trend/Dip Onayı)"
            elif in_position and longExit and i == N-1:
                current_bar_signal = "🛑 SAT Sinyali (Stop Patladı)"
                
            if current_bar_signal:
                bars_ago = (N - 1) - i
                ext = f" ({bars_ago} bar önce)" if bars_ago > 0 else ""
                latest_state = {
                    "signal": True,
                    "data": {
                        "type": current_bar_signal + ext, "gz_lower": gBot, "gz_upper": gTop,
                        "tp": aHigh + ((aHigh - aLow) * 0.618), "last_high": aHigh,
                        "stop": trailing_stop if not np.isnan(trailing_stop) else aLow
                    }
                }
                
    if not latest_state["signal"] and activeValid and aBull and not in_position:
        curr_close = closes[-1]
        dist_pct = (curr_close - gTop) / gTop
        if gBot <= curr_close <= gTop:
            latest_state = {"signal": True, "data": {"type": "⏳ Pusu Modu", "gz_lower": gBot, "gz_upper": gTop, "tp": aHigh + ((aHigh - aLow) * 0.618), "last_high": aHigh, "stop": trailing_stop if not np.isnan(trailing_stop) else aLow}}
        elif 0 < dist_pct <= 0.025:
            latest_state = {"signal": True, "data": {"type": f"👀 Yaklaşıyor (+%{dist_pct*100:.1f})", "gz_lower": gBot, "gz_upper": gTop, "tp": aHigh + ((aHigh - aLow) * 0.618), "last_high": aHigh, "stop": trailing_stop if not np.isnan(trailing_stop) else aLow}}

    # Portföydeki güncel durumu State'e kaydediyoruz
    latest_state["in_position"] = in_position

    return latest_state, trades

# =============================================================================
# 5. GRAFİK ÇİZİM MOTOLARI
# =============================================================================
def draw_gz_chart(symbol, df, ctx):
    plot_df = df.tail(100).copy()
    plt_subplots.style.use('dark_background')
    fig, ax = plt_subplots.subplots(figsize=(12, 6))
    
    up, down = plot_df['Close'] >= plot_df['Open'], plot_df['Close'] < plot_df['Open']
    ax.bar(plot_df.index[up], plot_df['Close'][up] - plot_df['Open'][up], 0.6, bottom=plot_df['Open'][up], color='#10b981')
    ax.bar(plot_df.index[down], plot_df['Open'][down] - plot_df['Close'][down], 0.6, bottom=plot_df['Close'][down], color='#ef4444')
    ax.vlines(plot_df.index[up], plot_df['Low'][up], plot_df['High'][up], color='#10b981', linewidth=1)
    ax.vlines(plot_df.index[down], plot_df['Low'][down], plot_df['High'][down], color='#ef4444', linewidth=1)
    
    if not np.isnan(ctx["gz_lower"]):
        ax.axhspan(ctx["gz_lower"], ctx["gz_upper"], color='#f59e0b', alpha=0.2, label='Altın Bölge (Golden Pocket)')
        ax.axhline(ctx["tp"], color='#00ffff', linestyle='--', linewidth=1.5, label='Kâr Al (1.618 Projeksiyon)')
    
    if not np.isnan(ctx["stop"]): ax.axhline(ctx["stop"], color='#ef4444', linestyle=':', linewidth=2.5, label='ATR İzleyen Stop Loss')
        
    ax.set_title(f"{symbol} | DURUM: {ctx['type']}", color='#e5e7eb', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper left', frameon=False, labelcolor='white')
    ax.grid(True, alpha=0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt_subplots.tight_layout()
    return fig

def draw_equity_curve(trades_df):
    plt_subplots.style.use('dark_background')
    fig, ax = plt_subplots.subplots(figsize=(14, 5))
    
    trades_df['Kümülatif Getiri (%)'] = trades_df['Kâr/Zarar (%)'].cumsum()
    
    ax.axhline(0, color='#ffffff', linestyle='-', alpha=0.3)
    color = '#10b981' if trades_df['Kümülatif Getiri (%)'].iloc[-1] >= 0 else '#ef4444'
    ax.plot(trades_df['Çıkış Tarihi'], trades_df['Kümülatif Getiri (%)'], color=color, linewidth=2.5, marker='o', markersize=5)
    ax.fill_between(trades_df['Çıkış Tarihi'], trades_df['Kümülatif Getiri (%)'], 0, color=color, alpha=0.15)
    
    ax.set_title("STRATEJİ KÜMÜLATİF GETİRİ EĞRİSİ (EQUITY CURVE)", color='#e5e7eb', fontsize=12, fontweight='bold')
    ax.set_ylabel("Net Getiri (%)", color='#9ca3af')
    ax.grid(True, alpha=0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt_subplots.tight_layout()
    return fig

# =============================================================================
# 6. ARAYÜZ (UI) MİMARİSİ (ÜÇLÜ MOTOR)
# =============================================================================
st.title("📈 GOLDEN ZONE WORKSTATION")

tab_scanner, tab_backtest, tab_bulk = st.tabs(["🚀 CANLI TARAYICI", "🔬 TEKLİ BACKTEST", "📊 TOPLU PİYASA BACKTESTİ"])

# -----------------------------------------------------------------------------
# TAB 1: CANLI TARAYICI
# -----------------------------------------------------------------------------
with tab_scanner:
    col_mkt, col_tf, col_btn = st.columns([2, 2, 1])
    with col_mkt: selected_mkt = st.selectbox("Piyasa Seçin:", list(MARKET_CONFIGS.keys()), key="mkt_scan")
    with col_tf: selected_tf = st.selectbox("Zaman Periyodu:", list(TIMEFRAME_CONFIGS.keys()), key="tf_scan")
    with col_btn: 
        st.write("##")
        execute_scan = st.button("TARAMAYI BAŞLAT", use_container_width=True)

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
            df_all = fetch_data_cached(yf_tickers, tf_config["period"], tf_config["interval"])
            
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
                else: continue
                    
                if tf_config["resample_rule"]:
                    try:
                        df_symbol.index = pd.to_datetime(df_symbol.index)
                        df_symbol = df_symbol.resample(tf_config["resample_rule"]).agg({
                            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                        }).dropna()
                    except Exception: continue
                    
                if len(df_symbol) < 50: continue
                    
                state, _ = core_pine_state_machine(df_symbol)
                
                if state["signal"]:
                    ctx = state["data"]
                    live_logs.append(f"[MATCH] {symbol:<6} : {ctx['type'].split('(')[0].strip()}")
                    console_placeholder.code("\n".join(live_logs[-15:]))
                    
                    curr_price = float(df_symbol['Close'].iloc[-1])
                    tv_interval = {"1h": "60", "2h": "120", "4h": "240", "1d": "D", "1wk": "W"}.get(tf_config["resample_rule"] or tf_config["interval"], "D")
                    tv_url = f"https://www.tradingview.com/chart/?symbol={mkt_config['tv_prefix']}{symbol}&interval={tv_interval}"
                    
                    found_signals.append({
                        "Durum": ctx["type"], "Hisse": symbol, "Fiyat": round(curr_price, 2),
                        "Giriş (Altın Bölge)": f"{ctx['gz_lower']:.2f} - {ctx['gz_upper']:.2f}",
                        "Kâr Hedefi": round(ctx["tp"], 2), "Stop Loss": round(ctx["stop"], 2), "Bağlantı": tv_url
                    })
                    stored_dfs[symbol] = {"df": df_symbol, "ctx": ctx}
            
            p_bar.empty()
            st.markdown(f"<div style='color:#10b981; font-family:Inter; font-weight:600;'>[SİSTEM BİLGİSİ] {selected_tf} Taraması tamamlandı.</div>", unsafe_allow_html=True)
            
            if found_signals:
                st.write("---"); st.write(f"### 🏆 ONAYLANMIŞ FIRSATLAR ({selected_tf})")
                res_df = pd.DataFrame(found_signals).sort_values(by="Durum", ascending=False)
                st.dataframe(res_df, use_container_width=True, hide_index=True, column_config={"Bağlantı": st.column_config.LinkColumn("TradingView", display_text="Grafiği Aç")})
                
                st.write("---"); st.write("### 🔬 GRAFİK İNCELEME İSTASYONU")
                selected_plot = st.selectbox("Detaylı inceleme için hisse seçin:", list(stored_dfs.keys()))
                if selected_plot: st.pyplot(draw_gz_chart(selected_plot, stored_dfs[selected_plot]["df"], stored_dfs[selected_plot]["ctx"]))
            else: st.warning(f"Bu periyotta ({selected_tf}) tespit edilen aktif bir durum bulunamadı.")

# -----------------------------------------------------------------------------
# TAB 2: TEKLİ BACKTEST LABORATUVARI
# -----------------------------------------------------------------------------
with tab_backtest:
    st.markdown("""
        <div style='background-color:#111827; padding:15px; border-left:4px solid #f59e0b; margin-bottom:20px;'>
            <div style='color:#e5e7eb; font-weight:600; font-size:14px; margin-bottom:5px;'>Tekil Kuantitatif Performans Laboratuvarı</div>
            <div style='color:#9ca3af; font-size:13px;'>Seçtiğiniz hissenin geçmiş verileri üzerinde algoritmayı çalıştırır ve detaylı al/sat noktalarını gösterir.</div>
        </div>
    """, unsafe_allow_html=True)

    col_mkt_bt, col_symbol_bt, col_tf_bt = st.columns([1, 1, 1])
    with col_mkt_bt: selected_mkt_bt = st.selectbox("Piyasa Seçin:", list(MARKET_CONFIGS.keys()), key="mkt_bt")
    mkt_config_bt = MARKET_CONFIGS[selected_mkt_bt]
    bt_symbols = get_all_market_symbols(mkt_config_bt)
    
    with col_symbol_bt: 
        if bt_symbols: target_symbol = st.selectbox("Hisse Sembolü Ara/Seç:", bt_symbols, key="sym_bt")
        else: target_symbol = st.text_input("Hisse Sembolü (Örn: THYAO, AAPL)", "THYAO", key="sym_bt_manual")
            
    with col_tf_bt: selected_tf_bt = st.selectbox("Zaman Periyodu:", list(TIMEFRAME_CONFIGS.keys()), key="tf_bt")

    tf_config_bt = TIMEFRAME_CONFIGS[selected_tf_bt]
    yf_ticker_bt = f"{target_symbol.replace('.', '-')}{mkt_config_bt['yf_suffix']}"
    auto_first_date = get_first_available_date(yf_ticker_bt, tf_config_bt["interval"])

    col_start, col_end, col_btn_bt = st.columns([1, 1, 1])
    with col_start: start_date = st.date_input("Başlangıç Tarihi", value=auto_first_date)
    with col_end: end_date = st.date_input("Bitiş Tarihi", value=datetime.today().date())
    with col_btn_bt: 
        st.write("##")
        run_backtest = st.button("ALGORİTMAYI TEST ET", use_container_width=True)

    if run_backtest:
        with st.spinner(f"{target_symbol} simüle ediliyor..."):
            df_bt = fetch_single_historical_data(yf_ticker_bt, tf_config_bt["interval"], start_date, end_date)
            
            if df_bt.empty: st.error("Veri bulunamadı.")
            else:
                if isinstance(df_bt.columns, pd.MultiIndex): df_bt.columns = df_bt.columns.get_level_values(0)
                    
                if tf_config_bt["resample_rule"]:
                    try:
                        df_bt.index = pd.to_datetime(df_bt.index)
                        df_bt = df_bt.resample(tf_config_bt["resample_rule"]).agg({
                            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                        }).dropna()
                    except Exception: pass
                
                if len(df_bt) < 50: st.warning("Yeterli veri yok.")
                else:
                    _, trades = core_pine_state_machine(df_bt)
                    
                    if not trades: st.info("İşlem gerçekleşmedi.")
                    else:
                        trades_df = pd.DataFrame(trades)
                        
                        total_trades = len(trades_df)
                        winning_trades = len(trades_df[trades_df['Kâr/Zarar (%)'] > 0])
                        win_rate = (winning_trades / total_trades) * 100
                        cumulative_return = trades_df['Kâr/Zarar (%)'].sum()
                        
                        max_drawdown, running_max = 0, 0
                        for val in trades_df['Kâr/Zarar (%)'].cumsum().values:
                            if val > running_max: running_max = val
                            if (running_max - val) > max_drawdown: max_drawdown = running_max - val
                            
                        st.write("---")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Toplam İşlem</div><div class='metric-value'>{total_trades}</div></div>", unsafe_allow_html=True)
                        with c2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Kazanma Oranı</div><div class='metric-value' style='color:#10b981;'>%{win_rate:.1f}</div></div>", unsafe_allow_html=True)
                        with c3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Kümülatif Net Getiri</div><div class='metric-value' style='color:{'#10b981' if cumulative_return>=0 else '#ef4444'};'>%{cumulative_return:.2f}</div></div>", unsafe_allow_html=True)
                        with c4: st.markdown(f"<div class='metric-card'><div class='metric-label'>Maksimum Düşüş</div><div class='metric-value' style='color:#ef4444;'>-%{max_drawdown:.2f}</div></div>", unsafe_allow_html=True)
                        
                        st.write("##")
                        st.pyplot(draw_equity_curve(trades_df))
                        st.write("### 📝 Detaylı İşlem Dökümü (Trade Log)")
                        st.dataframe(trades_df.style.map(lambda x: f"color: {'#10b981' if x > 0 else '#ef4444'}; font-weight: bold;", subset=['Kâr/Zarar (%)']), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 3: TOPLU PİYASA BACKTESTİ
# -----------------------------------------------------------------------------
with tab_bulk:
    st.markdown("""
        <div style='background-color:#111827; padding:15px; border-left:4px solid #8b5cf6; margin-bottom:20px;'>
            <div style='color:#e5e7eb; font-weight:600; font-size:14px; margin-bottom:5px;'>Market-Wide Backtest (Tüm Piyasa Simülasyonu)</div>
            <div style='color:#9ca3af; font-size:13px;'>Stratejinin seçilen periyotta ve piyasadaki tüm hisseler üzerindeki genel kazanma oranını ve performansını ölçer. Aynı zamanda her bir hissenin tam şu an hangi durumda olduğunu (Pozisyonda mı, Nakitte mi, Sinyal mi veriyor) gösterir.</div>
        </div>
    """, unsafe_allow_html=True)

    col_mkt_bulk, col_tf_bulk, col_btn_bulk = st.columns([2, 2, 1])
    with col_mkt_bulk: selected_mkt_bulk = st.selectbox("Piyasa Seçin:", list(MARKET_CONFIGS.keys()), key="mkt_bulk")
    with col_tf_bulk: selected_tf_bulk = st.selectbox("Zaman Periyodu:", list(TIMEFRAME_CONFIGS.keys()), key="tf_bulk")
    with col_btn_bulk: 
        st.write("##")
        run_bulk = st.button("TÜM PİYASAYI SİMÜLE ET", use_container_width=True)

    if run_bulk:
        mkt_config_bulk = MARKET_CONFIGS[selected_mkt_bulk]
        tf_config_bulk = TIMEFRAME_CONFIGS[selected_tf_bulk]
        
        market_symbols = get_all_market_symbols(mkt_config_bulk)
        if not market_symbols:
            st.error("Piyasa listesi alınamadı.")
        else:
            yf_tickers_bulk = [f"{s.replace('.', '-')}{mkt_config_bulk['yf_suffix']}" for s in market_symbols]
            
            with st.spinner(f"Toplu veri indiriliyor ve {len(market_symbols)} hisse analiz ediliyor..."):
                df_all_bulk = fetch_data_cached(yf_tickers_bulk, tf_config_bulk["period"], tf_config_bulk["interval"])
                
                bulk_results = []
                p_bar_bulk = st.progress(0)
                
                for idx, symbol in enumerate(market_symbols):
                    p_bar_bulk.progress((idx + 1) / len(market_symbols))
                    yf_ticker_key = f"{symbol.replace('.', '-')}{mkt_config_bulk['yf_suffix']}"
                    
                    if hasattr(df_all_bulk.columns, 'levels') and yf_ticker_key in df_all_bulk.columns.levels[0]:
                        df_symbol = df_all_bulk[yf_ticker_key].dropna(subset=['High', 'Close', 'Low', 'Open']).copy()
                    elif len(yf_tickers_bulk) == 1:
                        df_symbol = df_all_bulk.dropna(subset=['High', 'Close', 'Low', 'Open']).copy()
                    else: continue
                        
                    if tf_config_bulk["resample_rule"]:
                        try:
                            df_symbol.index = pd.to_datetime(df_symbol.index)
                            df_symbol = df_symbol.resample(tf_config_bulk["resample_rule"]).agg({
                                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                            }).dropna()
                        except Exception: continue
                        
                    if len(df_symbol) < 50: continue
                        
                    state, trades = core_pine_state_machine(df_symbol)
                    
                    # Güncel Durum Saptaması
                    if state["signal"]:
                        guncel_durum = state["data"]["type"].split('(')[0].strip()
                    else:
                        guncel_durum = "🟢 Pozisyonda (Trend Devam)" if state.get("in_position") else "🔴 Nakitte (Fırsat Bekleniyor)"

                    if trades:
                        trades_df = pd.DataFrame(trades)
                        total = len(trades_df)
                        wins = len(trades_df[trades_df['Kâr/Zarar (%)'] > 0])
                        win_rate = (wins / total) * 100
                        net_return = trades_df['Kâr/Zarar (%)'].sum()
                        
                        max_drawdown, running_max = 0, 0
                        for val in trades_df['Kâr/Zarar (%)'].cumsum().values:
                            if val > running_max: running_max = val
                            if (running_max - val) > max_drawdown: max_drawdown = running_max - val
                            
                        bulk_results.append({
                            "Hisse": symbol,
                            "Güncel Durum": guncel_durum,
                            "Toplam İşlem": total,
                            "Kazanma Oranı (%)": round(win_rate, 2),
                            "Net Getiri (%)": round(net_return, 2),
                            "Maksimum Düşüş (%)": round(-max_drawdown, 2)
                        })
                
                p_bar_bulk.empty()
                
                if bulk_results:
                    st.success(f"{len(bulk_results)} adet hissenin backtest işlemi tamamlandı.")
                    bulk_df = pd.DataFrame(bulk_results)
                    
                    bulk_df = bulk_df.sort_values(by="Net Getiri (%)", ascending=False)
                    
                    def style_status(val):
                        if "🎯" in val or "🔥" in val or "📈" in val: return "color: #10b981; font-weight: bold;"
                        elif "🟢" in val: return "color: #3b82f6;"
                        elif "🛑" in val: return "color: #ef4444; font-weight: bold;"
                        elif "⏳" in val or "👀" in val: return "color: #f59e0b;"
                        else: return "color: #ef4444;"
                        
                    st.dataframe(
                        bulk_df.style.map(style_status, subset=['Güncel Durum'])
                                     .map(lambda x: f"color: {'#10b981' if x > 0 else '#ef4444'}; font-weight: bold;", subset=['Net Getiri (%)'])
                                     .map(lambda x: f"color: {'#10b981' if x >= 60 else '#f59e0b' if x >= 40 else '#ef4444'};", subset=['Kazanma Oranı (%)']),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.warning("Bu periyotta işlem gerçekleşen hisse bulunamadı.")
