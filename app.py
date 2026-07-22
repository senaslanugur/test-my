import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Golden Zone Tarayıcı", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="darkgrid")

# --- PARAMETRELER VE LİSTELER ---
BIST_SEMBOLLER = ["THYAO.IS", "KCHOL.IS", "TUPRS.IS", "AKBNK.IS", "ISCTR.IS", "SAHOL.IS", "ASELS.IS", "SISE.IS", "BIMAS.IS", "YKBNK.IS"] # Genişletilebilir
US_SEMBOLLER = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "INTC"] # Genişletilebilir

TIMEFRAME_MAP = {
    "1 Saat": "1h",
    "2 Saat": "1h", # yfinance'de 2h stabil çalışmadığı için 1h çekip resample edilebilir, basitleştirmek için 1h kullanıldı.
    "4 Saat": "1d", # Demo amaçlı, yf limitleri için.
    "1 Gün": "1d",
    "1 Hafta": "1wk"
}

# --- YARDIMCI FONKSİYONLAR (ALGORİTMA) ---
def get_pivots(df, pivot_len=15, confirm_bars=5):
    """Basitleştirilmiş ZigZag / Pivot noktası tespiti"""
    highs = df['High'].values
    lows = df['Low'].values
    
    ph = np.full(len(df), np.nan)
    pl = np.full(len(df), np.nan)
    
    for i in range(pivot_len, len(df) - confirm_bars):
        window_high = highs[i - pivot_len : i + confirm_bars + 1]
        window_low = lows[i - pivot_len : i + confirm_bars + 1]
        
        if highs[i] == np.max(window_high):
            ph[i] = highs[i]
        if lows[i] == np.min(window_low):
            pl[i] = lows[i]
            
    return ph, pl

def calculate_golden_zone(df):
    """Pine Script'teki f_retracement_price ve f_projection_price mantığı"""
    ph, pl = get_pivots(df)
    
    # Son tepe ve dibi bul
    last_high_idx = np.where(~np.isnan(ph))[0]
    last_low_idx = np.where(~np.isnan(pl))[0]
    
    if len(last_high_idx) == 0 or len(last_low_idx) == 0:
        return None
        
    last_high = ph[last_high_idx[-1]]
    last_low = pl[last_low_idx[-1]]
    
    # Yükseliş dalgası (Bullish Leg) varsayımı ile Golden Zone (0.5 - 0.618)
    range_val = last_high - last_low
    gz_upper = last_high - (range_val * 0.5)
    gz_lower = last_high - (range_val * 0.618)
    
    # Kâr Al (Take Profit) - 1.618 Projection
    tp_1618 = last_high + (range_val * 0.618)
    
    # Sinyal Kontrolü (Tam Al Noktası: Fiyat GZ içine girip yukarı çıkmış mı?)
    current_close = df['Close'].iloc[-1]
    current_low = df['Low'].iloc[-1]
    
    # Fitil golden zone'a değmiş, ancak kapanış golden zone'un üzerinde (Bull Rejection)
    is_signal = (current_low <= gz_upper) and (current_low >= gz_lower) and (current_close > gz_upper)
    
    return {
        "signal": is_signal,
        "gz_lower": gz_lower,
        "gz_upper": gz_upper,
        "tp": tp_1618,
        "last_high": last_high,
        "last_low": last_low,
        "last_high_idx": last_high_idx[-1],
        "last_low_idx": last_low_idx[-1]
    }

def analyze_ticker(ticker, interval):
    """Bireysel hisse verisini çeker ve analiz eder"""
    try:
        # Intraday periyotlar için limitli gün çekilmeli
        period = "1mo" if interval in ["1h", "90m"] else "1y"
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty or len(df) < 30:
            return None
            
        result = calculate_golden_zone(df)
        if result and result["signal"]:
            result["ticker"] = ticker
            result["df"] = df
            result["current_price"] = df['Close'].iloc[-1]
            return result
    except Exception:
        pass
    return None

# --- GÖRSELLEŞTİRME (MATPLOTLIB / SEABORN) ---
def plot_setup(result):
    """Sinyal veren hissenin Matplotlib/Seaborn grafiğini oluşturur"""
    df = result["df"]
    ticker = result["ticker"]
    
    # Son 50 mumu göster
    plot_df = df.iloc[-50:].copy()
    plot_df.reset_index(inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Kapanış fiyatı çizgisi
    sns.lineplot(data=plot_df, x=plot_df.index, y='Close', color='white', linewidth=2, ax=ax)
    
    # Golden Zone Alanı Boyama
    ax.axhspan(result["gz_lower"], result["gz_upper"], color='green', alpha=0.3, label='Golden Zone (0.5 - 0.618)')
    
    # Kâr Al (TP) Çizgisi
    ax.axhline(result["tp"], color='gold', linestyle='--', linewidth=1.5, label='Kâr Al (1.618)')
    
    ax.set_title(f"{ticker} - Golden Zone Alım Fırsatı", color='white')
    ax.set_xlabel("Son Barlar")
    ax.set_ylabel("Fiyat")
    ax.legend(loc='upper left', facecolor='black', labelcolor='white')
    
    # Arka plan ve estetik dokunuşlar
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444444')
        
    return fig

# --- ARAYÜZ (UI) ---
st.title("📈 HexaTrades: Fibonacci Golden Zone Kuantitatif Tarayıcı")
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Tarama Kriterleri")
    market_selection = st.selectbox("Piyasa Seçimi", ["BİST (Örnek Liste)", "ABD Borsası (Örnek Liste)"])
    timeframe_selection = st.selectbox("Periyot", list(TIMEFRAME_MAP.keys()))
    scan_button = st.button("🚀 Piyasayı Tara", use_container_width=True)

with col2:
    if scan_button:
        interval = TIMEFRAME_MAP[timeframe_selection]
        tickers = BIST_SEMBOLLER if "BİST" in market_selection else US_SEMBOLLER
        
        st.info(f"**{market_selection}** piyasasında **{timeframe_selection}** periyot için güncel fiyatlar taranıyor... Bu işlem birkaç saniye sürebilir.")
        
        progress_bar = st.progress(0)
        signals = []
        
        # Çoklu thread ile asenkron tarama (Hız optimizasyonu)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(analyze_ticker, t, interval): t for t in tickers}
            for i, future in enumerate(futures):
                res = future.result()
                if res:
                    signals.append(res)
                progress_bar.progress((i + 1) / len(tickers))
                
        progress_bar.empty()
        
        if not signals:
            st.warning("Şu anda tam al noktasında (Golden Zone içinde reaksiyon veren) bir hisse bulunamadı.")
        else:
            st.success(f"Tam al noktasında {len(signals)} adet hisse tespit edildi!")
            
            for res in signals:
                ticker = res["ticker"]
                current_p = float(res["current_price"])
                tp_p = float(res["tp"])
                kar_potansiyeli = ((tp_p - current_p) / current_p) * 100
                
                # TradingView Linki Oluşturma
                tv_prefix = "BIST:" if "BİST" in market_selection else "NASDAQ:"
                tv_link = f"https://www.tradingview.com/chart/?symbol={tv_prefix}{ticker.replace('.IS', '')}"
                
                with st.expander(f"🟢 {ticker} | Fiyat: {current_p:.2f} | Hedef: {tp_p:.2f} (+%{kar_potansiyeli:.2f})", expanded=True):
                    st.markdown(f"**[🔍 TradingView'da İncele]({tv_link})**")
                    
                    # Grafiği çiz
                    fig = plot_setup(res)
                    st.pyplot(fig)


