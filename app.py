import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# --- SAYFA VE GRAFİK KONFİGÜRASYONU ---
st.set_page_config(page_title="Golden Zone Tarayıcı", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="darkgrid")

# --- PİYASA VE ZAMAN PERİYODU AYARLARI ---
MARKET_CONFIGS = {
    "Türkiye (BIST)": {"tv_market": "turkey", "yf_suffix": ".IS", "tv_prefix": "BIST:"},
    "Amerika (ABD)": {"tv_market": "america", "yf_suffix": "", "tv_prefix": ""}
}

TIMEFRAME_MAP = {
    "1 Saat": "1h",
    "2 Saat": "1h", 
    "4 Saat": "1d", 
    "1 Gün": "1d",
    "1 Hafta": "1wk"
}

# --- TRADINGVIEW API İLE LİSTE ÇEKME ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_all_market_symbols(mkt_config):
    """TradingView tarayıcısından piyasadaki hisse sembollerini çeker."""
    url = f"https://scanner.tradingview.com/{mkt_config['tv_market']}/scan"
    payload = {
        "filter": [{"left": "type", "operation": "in_range", "right": ["stock"]}],
        "options": {"lang": "en"}, 
        "markets": [mkt_config['tv_market']],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "market_cap_basic"],
        "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"}, 
        "range": [0, 1000] # Piyasa değerine göre en büyük 1000 hisseyi çeker
    }
    
    try:
        resp = requests.post(url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code == 200:
            return [item["d"][0] for item in resp.json().get("data", [])]
    except Exception:
        pass
    return []

# --- PINE SCRIPT BİREBİR MATEMATİKSEL ENTEGRASYONU ---
def calculate_atr(high, low, close, period=14):
    """Pine Script'teki ta.atr (RMA bazlı True Range) hesaplaması"""
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.insert(tr, 0, high[0] - low[0])
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    return atr

def evaluate_golden_zone_pine_mimic(df, signal_window=3):
    """
    Pine Script v6 mantığını bar-bar iteratif olarak simüle eden çekirdek motor.
    signal_window: Sinyalin son kaç bar içinde gerçekleştiğini tolere edeceğimiz değer.
    """
    pivotLen = 15
    confirmBars = 5
    zzDevAtr = 1.5
    invBufAtr = 0.3
    goldenLower = 0.5
    goldenUpper = 0.618
    
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    open_ = df['Open'].values
    
    atr = calculate_atr(high, low, close, 14)
    N = len(high)
    
    zzP1, zzP0 = np.nan, np.nan
    zzD1 = 0
    zzLow, zzHigh = np.nan, np.nan
    
    aSet, aAlive = False, False
    aHigh, aLow = np.nan, np.nan
    gTop, gBot = np.nan, np.nan
    aRejected = False
    trailingStop = np.nan
    
    signal_triggered = False
    signal_type = ""
    bars_since_signal = 0
    
    if N < pivotLen + confirmBars:
        return None
        
    for i in range(pivotLen + confirmBars, N):
        isZigZagHigh = False
        isZigZagLow = False
        zzLegEvent = False
        
        idx_eval = i - confirmBars
        window_high = high[idx_eval - pivotLen : i + 1]
        window_low = low[idx_eval - pivotLen : i + 1]
        
        usePH = np.nan
        usePL = np.nan
        
        if high[idx_eval] == np.max(window_high):
            usePH = high[idx_eval]
        if low[idx_eval] == np.min(window_low):
            usePL = low[idx_eval]
            
        if not np.isnan(usePH) and not np.isnan(usePL):
            if np.isnan(zzP1):
                usePH = np.nan; usePL = np.nan
            else:
                dH = abs(usePH - zzP1)
                dL = abs(usePL - zzP1)
                if dH > dL: usePL = np.nan
                elif dL > dH: usePH = np.nan
                else: usePH = np.nan; usePL = np.nan
        
        pivotAtr = atr[idx_eval] if not np.isnan(atr[idx_eval]) else 0
        zzMinLeg = zzDevAtr * pivotAtr
        
        if not np.isnan(usePH):
            if zzD1 == 1:
                if usePH > zzP1:
                    zzP1 = usePH; zzHigh = usePH
                    zzLegEvent = True; isZigZagHigh = True
            elif np.isnan(zzP1):
                zzP1 = usePH; zzD1 = 1; zzHigh = usePH; isZigZagHigh = True
            elif abs(usePH - zzP1) > zzMinLeg:
                zzP0 = zzP1; zzP1 = usePH; zzD1 = 1; zzHigh = usePH
                zzLegEvent = True; isZigZagHigh = True
                
        if not np.isnan(usePL):
            if zzD1 == -1:
                if usePL < zzP1:
                    zzP1 = usePL; zzLow = usePL
                    zzLegEvent = True; isZigZagLow = True
            elif np.isnan(zzP1):
                zzP1 = usePL; zzD1 = -1; zzLow = usePL; isZigZagLow = True
            elif abs(usePL - zzP1) > zzMinLeg:
                zzP0 = zzP1; zzP1 = usePL; zzD1 = -1; zzLow = usePL
                zzLegEvent = True; isZigZagLow = True
                
        if isZigZagLow and not np.isnan(zzLow):
            trailingStop = zzLow - (invBufAtr * atr[i])
            
        validLeg = (zzD1 != 0) and not np.isnan(zzP0) and not np.isnan(zzP1) and (zzP0 != zzP1)
        dirBull = (zzD1 == 1)
        legHigh = max(zzP0, zzP1) if validLeg else np.nan
        legLow = min(zzP0, zzP1) if validLeg else np.nan
        
        validSetup = validLeg and (legHigh > legLow)
        candidateEvent = zzLegEvent and validSetup
        
        if candidateEvent:
            aSet = True; aAlive = True
            aHigh = legHigh; aLow = legLow
            aRejected = False
            
            rng = aHigh - aLow
            gTop = aHigh - (goldenLower * rng)
            gBot = aHigh - (goldenUpper * rng)
            
        activeValid = aSet and aAlive and not np.isnan(aHigh) and not np.isnan(aLow) and (aHigh - aLow) > 0
        evBullRej = False
        
        if activeValid and dirBull:
            touchWick = (low[i] <= gTop)
            bullRejectRaw = touchWick and (close[i] > gTop) and (close[i] > open_[i])
            
            if bullRejectRaw and not aRejected:
                aRejected = True
                evBullRej = True
                
        longEnter = (evBullRej or isZigZagLow) and activeValid and dirBull
        
        # SİNYAL PENCERESİ ESNEMESİ (Son 'signal_window' kadar barı kontrol et)
        if i >= N - signal_window:
            if longEnter:
                signal_triggered = True
                bars_since_signal = (N - 1) - i
                signal_type = "Golden Zone Temas & Red" if evBullRej else "ZigZag Yeni Dip Onayı"

    if signal_triggered:
        tp_1618 = aHigh + (aHigh - aLow) * 0.618
        
        # Eğer sinyal geçmiş barlarda geldiyse, ekrana bilgi olarak yazdır
        if bars_since_signal > 0:
            signal_type += f" ({bars_since_signal} bar önce)"
            
        return {
            "signal": True,
            "signal_type": signal_type,
            "gz_lower": gBot,
            "gz_upper": gTop,
            "tp": tp_1618,
            "last_high": aHigh,
            "last_low": aLow,
            "stop_loss": trailingStop if not np.isnan(trailingStop) else aLow
        }
    return None



def analyze_ticker(symbol, mkt_config, interval):
    clean_symbol = symbol.replace('.', '-')
    yf_ticker = f"{clean_symbol}{mkt_config['yf_suffix']}"
    
    try:
        period = "3mo" if interval in ["1h", "90m"] else "2y"
        df = yf.download(tickers=yf_ticker, period=period, interval=interval, progress=False, show_errors=False)
        
        if df.empty or len(df) < 50:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        result = evaluate_golden_zone_pine_mimic(df)
        if result and result["signal"]:
            result["ticker"] = symbol
            result["df"] = df
            result["current_price"] = df['Close'].iloc[-1]
            return result
    except Exception:
        pass
    return None

# --- GÖRSELLEŞTİRME ---
def plot_setup(result):
    df = result["df"]
    ticker = result["ticker"]
    
    plot_df = df.iloc[-100:].copy()
    plot_df.reset_index(inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(data=plot_df, x=plot_df.index, y='Close', color='#00E676', linewidth=2, ax=ax)
    
    ax.axhspan(result["gz_lower"], result["gz_upper"], color='#388E3C', alpha=0.3, label='Golden Zone (0.5 - 0.618)')
    ax.axhline(result["tp"], color='#FFD600', linestyle='--', linewidth=1.5, label='Kâr Al Hedefi (1.618)')
    ax.axhline(result["stop_loss"], color='#D32F2F', linestyle=':', linewidth=2, label='ATR İzleyen Stop (Trailing)')
    
    ax.set_title(f"{ticker} - Sinyal: {result['signal_type']}", color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel("Son Barlar", color='lightgray')
    ax.set_ylabel("Fiyat", color='lightgray')
    ax.legend(loc='upper left', facecolor='#212121', labelcolor='white', framealpha=0.8)
    
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    ax.tick_params(colors='lightgray')
    for spine in ax.spines.values():
        spine.set_color('#333333')
        
    return fig

# --- ARAYÜZ (UI) ---
st.title("📊 Tam Kapsamlı Golden Zone Tarayıcı")
st.markdown("Piyasadaki hisseler **TradingView altyapısı** üzerinden anlık taranarak Altın Bölge'de olanlar listelenir.")
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Tarama Kriterleri")
    market_selection = st.selectbox("Piyasa Seçimi:", list(MARKET_CONFIGS.keys()))
    timeframe_selection = st.selectbox("Periyot:", list(TIMEFRAME_MAP.keys()))
    scan_button = st.button("🚀 Derin Taramayı Başlat", use_container_width=True)

with col2:
    if scan_button:
        interval = TIMEFRAME_MAP[timeframe_selection]
        mkt_config = MARKET_CONFIGS[market_selection]
        
        st.info("TradingView veritabanından güncel hisse listesi çekiliyor...")
        
        tv_symbols = get_all_market_symbols(mkt_config)
            
        if not tv_symbols:
            st.error("TradingView'den hisse listesi alınamadı. Lütfen ağ bağlantınızı kontrol edin.")
            st.stop()
            
        st.success(f"Başarılı! TradingView üzerinden **{len(tv_symbols)}** adet hisse bulundu ve taranıyor. Lütfen sekmeyi kapatmayın.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        signals = []
        
        completed = 0
        total_tickers = len(tv_symbols)
        
        # Paralel İşleme
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_ticker = {executor.submit(analyze_ticker, t, mkt_config, interval): t for t in tv_symbols}
            
            for future in as_completed(future_to_ticker):
                res = future.result()
                if res:
                    signals.append(res)
                    
                completed += 1
                if completed % 5 == 0 or completed == total_tickers:
                    progress_bar.progress(completed / total_tickers)
                    status_text.text(f"İşlenen: {completed} / {total_tickers} | Bulunan Sinyal: {len(signals)}")
                
        progress_bar.empty()
        status_text.empty()
        
        st.markdown("---")
        if not signals:
            st.warning("Bu periyotta Altın Bölge'ye (Golden Zone) tam temas edip tepki alan hisse bulunamadı.")
        else:
            st.success(f"🎯 Tam al noktasında **{len(signals)}** adet hisse yakalandı!")
            
            for res in signals:
                ticker = res["ticker"]
                
                current_p = float(res["current_price"].iloc[0]) if isinstance(res["current_price"], pd.Series) else float(res["current_price"])
                tp_p = float(res["tp"])
                kar_potansiyeli = ((tp_p - current_p) / current_p) * 100
                
                tv_link = f"https://www.tradingview.com/chart/?symbol={mkt_config['tv_prefix']}{ticker}"
                
                with st.expander(f"🟢 {ticker} | Mevcut Fiyat: {current_p:.2f} | Kâr Al: {tp_p:.2f} (+%{kar_potansiyeli:.2f})", expanded=False):
                    col_info, col_chart = st.columns([1, 4])
                    
                    with col_info:
                        st.markdown(f"**Sembol:** {ticker}")
                        st.markdown(f"**Giriş Bandı:** {res['gz_lower']:.2f} - {res['gz_upper']:.2f}")
                        st.markdown(f"**Stop (ATR):** {res['stop_loss']:.2f}")
                        st.markdown(f"**Sinyal:** {res['signal_type']}")
                        st.markdown(f"[📊 TradingView'da Aç]({tv_link})")
                        
                    with col_chart:
                        fig = plot_setup(res)
                        st.pyplot(fig)
                        plt.close(fig)
