import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import yahoo_fin.stock_info as si

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Golden Zone Tarayıcı", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="darkgrid")

# --- ZAMAN DİLİMLERİ ---
TIMEFRAME_MAP = {
    "1 Saat": "1h",
    "2 Saat": "1h", # yfinance 2h stabil desteklemediğinden 1h kullanıldı
    "4 Saat": "1d", # Limitler gereği günlük
    "1 Gün": "1d",
    "1 Hafta": "1wk"
}

# --- DİNAMİK HİSSE LİSTESİ ÇEKİCİLER ---

@st.cache_data(ttl=86400) # Veriyi 24 saat önbellekte tut
def get_all_bist_tickers():
    """İş Yatırım üzerinden güncel tüm BİST hisselerini çeker."""
    try:
        url = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/Common/Data.aspx/HisseGetir"
        response = requests.get(url, timeout=10).json()
        tickers = [f"{item['kod']}.IS" for item in response.get('veri', [])]
        return tickers
    except Exception as e:
        st.error("BİST hisseleri çekilemedi, lütfen internet bağlantınızı kontrol edin.")
        return []

@st.cache_data(ttl=86400)
def get_all_us_tickers():
    """yahoo_fin kullanarak tüm NASDAQ ve NYSE hisselerini çeker."""
    try:
        nasdaq = si.tickers_nasdaq()
        nyse = si.tickers_nyse()
        # Birleştir ve tekrarları kaldır, yfinance formatına uygun hale getir
        tickers = list(set(nasdaq + nyse))
        clean_tickers = [str(t).replace("^", "-") for t in tickers if not pd.isna(t)]
        return clean_tickers
    except Exception as e:
        st.error("ABD hisseleri çekilemedi.")
        return []

# --- YARDIMCI FONKSİYONLAR (ALGORİTMA) ---
def get_pivots(df, pivot_len=15, confirm_bars=5):
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
    ph, pl = get_pivots(df)
    
    last_high_idx = np.where(~np.isnan(ph))[0]
    last_low_idx = np.where(~np.isnan(pl))[0]
    
    if len(last_high_idx) == 0 or len(last_low_idx) == 0:
        return None
        
    last_high = ph[last_high_idx[-1]]
    last_low = pl[last_low_idx[-1]]
    
    range_val = last_high - last_low
    gz_upper = last_high - (range_val * 0.5)
    gz_lower = last_high - (range_val * 0.618)
    
    tp_1618 = last_high + (range_val * 0.618)
    
    current_close = df['Close'].iloc[-1]
    current_low = df['Low'].iloc[-1]
    
    # Tam Al Noktası: Fiyat GZ içine fitil atmış ancak kapanış GZ üzerinde
    is_signal = (current_low <= gz_upper) and (current_low >= gz_lower) and (current_close > gz_upper)
    
    return {
        "signal": is_signal,
        "gz_lower": gz_lower,
        "gz_upper": gz_upper,
        "tp": tp_1618,
        "last_high": last_high,
        "last_low": last_low
    }

def analyze_ticker(ticker, interval):
    try:
        period = "1mo" if interval in ["1h", "90m"] else "2y"
        df = yf.download(ticker, period=period, interval=interval, progress=False, show_errors=False)
        
        # Hacimsiz, hatalı veya yeni halka arz olmuş yetersiz verili hisseleri atla
        if df.empty or len(df) < 50:
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

# --- GÖRSELLEŞTİRME ---
def plot_setup(result):
    df = result["df"]
    ticker = result["ticker"]
    
    plot_df = df.iloc[-80:].copy()
    plot_df.reset_index(inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(data=plot_df, x=plot_df.index, y='Close', color='#00E676', linewidth=2, ax=ax)
    
    ax.axhspan(result["gz_lower"], result["gz_upper"], color='#388E3C', alpha=0.3, label='Golden Zone (0.5 - 0.618)')
    ax.axhline(result["tp"], color='#FFD600', linestyle='--', linewidth=1.5, label='Kâr Al Hedefi (1.618)')
    ax.axhline(result["last_low"], color='#D32F2F', linestyle=':', linewidth=1.5, label='Stop / Son Dip')
    
    ax.set_title(f"{ticker} - Altın Bölge Reaksiyonu", color='white', fontsize=14, fontweight='bold')
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
st.markdown("Piyasadaki **tüm aktif hisseler** taranarak Altın Bölge'de olanlar listelenir.")
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Tarama Kriterleri")
    market_selection = st.selectbox("Piyasa Seçimi", ["Tüm BİST Hisseleri", "Tüm ABD Hisseleri (NASDAQ & NYSE)"])
    timeframe_selection = st.selectbox("Periyot", list(TIMEFRAME_MAP.keys()))
    scan_button = st.button("🚀 Derin Taramayı Başlat", use_container_width=True)

with col2:
    if scan_button:
        interval = TIMEFRAME_MAP[timeframe_selection]
        
        st.info("Borsa veritabanından güncel hisse listesi çekiliyor...")
        if "BİST" in market_selection:
            tickers = get_all_bist_tickers()
        else:
            tickers = get_all_us_tickers()
            
        if not tickers:
            st.error("Hisse listesi alınamadı. İşlemi iptal ediyorum.")
            st.stop()
            
        st.success(f"Başarılı! Toplam **{len(tickers)}** adet hisse taranıyor. Bu işlem çok sayıda hisse içerdiği için biraz zaman alabilir, lütfen sekmeyi kapatmayın.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        signals = []
        
        completed = 0
        total_tickers = len(tickers)
        
        # Paralel İşleme - Çoklu İstek Gönderimi
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_ticker = {executor.submit(analyze_ticker, t, interval): t for t in tickers}
            
            for future in as_completed(future_to_ticker):
                res = future.result()
                if res:
                    signals.append(res)
                    
                completed += 1
                # Progress bar'ı yormamak için her 5 hissedede bir güncelle
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
                
                # Tek boyutlu numpy serisi (yfinance versiyon farklılıkları için güvenlik)
                current_p = float(res["current_price"].iloc[0]) if isinstance(res["current_price"], pd.Series) else float(res["current_price"])
                tp_p = float(res["tp"])
                kar_potansiyeli = ((tp_p - current_p) / current_p) * 100
                
                tv_prefix = "BIST:" if "BİST" in market_selection else ""
                clean_ticker_for_tv = ticker.replace('.IS', '')
                tv_link = f"https://www.tradingview.com/chart/?symbol={tv_prefix}{clean_ticker_for_tv}"
                
                with st.expander(f"🟢 {clean_ticker_for_tv} | Mevcut Fiyat: {current_p:.2f} | Kâr Al: {tp_p:.2f} (+%{kar_potansiyeli:.2f})", expanded=False):
                    col_info, col_chart = st.columns([1, 4])
                    
                    with col_info:
                        st.markdown(f"**Sembol:** {clean_ticker_for_tv}")
                        st.markdown(f"**Giriş Bandı:** {res['gz_lower']:.2f} - {res['gz_upper']:.2f}")
                        st.markdown(f"**Stop (Son Dip):** {res['last_low']:.2f}")
                        st.markdown(f"[📊 TradingView'da Aç]({tv_link})")
                        
                    with col_chart:
                        fig = plot_setup(res)
                        st.pyplot(fig)
                        plt.close(fig) # Bellek sızıntısını önlemek için grafiği temizle
