import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ==========================================
# 1. SAYFA VE ARAYÜZ KONFİGÜRASYONU
# ==========================================
st.set_page_config(page_title="Golden Zone Kuantitatif Tarayıcı", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="darkgrid")

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

# ==========================================
# 2. TRADINGVIEW VERİ ÇEKME MOTORU
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_all_market_symbols(mkt_config):
    """TradingView üzerinden piyasadaki hisseleri hacme göre filtreleyerek çeker."""
    url = f"https://scanner.tradingview.com/{mkt_config['tv_market']}/scan"
    payload = {
        "filter": [{"left": "type", "operation": "in_range", "right": ["stock"]}],
        "options": {"lang": "en"}, 
        "markets": [mkt_config['tv_market']],
        "symbols": {"query": {"types": []}, "tickers": []},
        "columns": ["name", "volume"],
        "sort": {"sortBy": "volume", "sortOrder": "desc"}, 
        "range": [0, 800] 
    }
    try:
        resp = requests.post(url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if resp.status_code == 200:
            return [item["d"][0] for item in resp.json().get("data", [])]
    except Exception:
        pass
    return []

# ==========================================
# 3. YENİ NESİL VEKTÖREL ALGORİTMA MOTORU
# ==========================================
def find_golden_zone_vectorized(df):
    """
    Kırılgan döngüler yerine, istatistiksel Swing High/Low noktalarını 
    vektörel olarak bulan ve Golden Zone'u hesaplayan sağlam mimari.
    """
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    
    pivots = []
    # 15 bar geriye, 5 bar ileriye bakarak mutlak tepe ve dipleri bul
    for i in range(15, len(df) - 5):
        if highs[i] == np.max(highs[i-15 : i+6]):
            pivots.append((i, 'H', highs[i]))
        if lows[i] == np.min(lows[i-15 : i+6]):
            pivots.append((i, 'L', lows[i]))
            
    if len(pivots) < 2: 
        return None
        
    # Arka arkaya gelen aynı yönlü tepe/dipleri filtrele (ZigZag sadeliği)
    zz = [pivots[0]]
    for p in pivots[1:]:
        last_p = zz[-1]
        if p[1] == last_p[1]:
            if p[1] == 'H' and p[2] > last_p[2]: zz[-1] = p
            elif p[1] == 'L' and p[2] < last_p[2]: zz[-1] = p
        else:
            zz.append(p)
            
    if len(zz) < 2: 
        return None
        
    last_pivot = zz[-1]
    prev_pivot = zz[-2]
    
    # KURAL 1: Yükseliş yönlü (Bullish) bir yapı için, son oluşan pivot TEPE (High) olmalıdır.
    if last_pivot[1] != 'H': 
        return None
        
    leg_high = last_pivot[2]
    leg_low = prev_pivot[2]
    
    if leg_high <= leg_low: 
        return None
        
    # KURAL 2: Fibonacci Bölgesi Hesaplama
    rng = leg_high - leg_low
    gz_upper = leg_high - (0.5 * rng)
    gz_lower = leg_high - (0.618 * rng)
    
    # Son tepe oluştuktan sonraki fiyat hareketleri (Geri Çekilme Evresi)
    pullback_bars = df.iloc[last_pivot[0] + 1 : ]
    if pullback_bars.empty: 
        return None
        
    touched_zone = pullback_bars['Low'].min() <= gz_upper
    
    curr_close = closes[-1]
    curr_low = lows[-1]
    curr_open = opens[-1]
    
    signal = False
    signal_type = ""
    
    # SİNYAL 1: Kusursuz Pine Script Ret İşlemi (Fitil atıp üstünde kapattı)
    if curr_low <= gz_upper and curr_close > gz_upper and curr_close > curr_open:
        signal = True
        signal_type = "🎯 Kusursuz Ret (Bullish Rejection)"
        
    # SİNYAL 2: Radar/Pusu Modu (Fiyat şu an tam bölgenin içinde tepki bekliyor)
    elif gz_lower <= curr_close <= gz_upper:
        signal = True
        signal_type = "⏳ Pusu Modu: Fiyat Altın Bölge İçinde"
        
    # SİNYAL 3: Güvenli Alım (Bölgeye değdi ve yukarı doğru kırılım başladı)
    elif touched_zone and curr_close > gz_upper and curr_close > curr_open:
        signal = True
        signal_type = "🚀 Bölgeden Onaylı Çıkış"
        
    if signal:
        tp_1618 = leg_high + (rng * 0.618)
        # Zarar Kes (Stop Loss), başladığı yükseliş dibinin bir tık altıdır
        stop_loss = leg_low * 0.99
        
        return {
            "signal": True,
            "signal_type": signal_type,
            "gz_lower": gz_lower,
            "gz_upper": gz_upper,
            "tp": tp_1618,
            "last_high": leg_high,
            "last_low": leg_low,
            "stop_loss": stop_loss
        }
        
    return None

def analyze_ticker(symbol, mkt_config, interval):
    clean_symbol = symbol.replace('.', '-')
    yf_ticker = f"{clean_symbol}{mkt_config['yf_suffix']}"
    
    try:
        # Yfinance veri limiti kuralları ihlal edilmeden en geniş veriyi çek
        if interval in ["1h", "90m"]: period = "1mo"
        elif interval == "1d": period = "1y"
        else: period = "2y"
        
        df = yf.download(tickers=yf_ticker, period=period, interval=interval, progress=False, show_errors=False)
        
        if df.empty or len(df) < 50:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        result = find_golden_zone_vectorized(df)
        if result and result["signal"]:
            result["ticker"] = symbol
            result["df"] = df
            result["current_price"] = df['Close'].iloc[-1]
            return result
    except Exception:
        pass
    return None

# ==========================================
# 4. PROFESYONEL GRAFİK ÇİZİM MOTORU
# ==========================================
def plot_setup(result):
    df = result["df"]
    ticker = result["ticker"]
    
    plot_df = df.iloc[-90:].copy()
    plot_df.reset_index(inplace=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.lineplot(data=plot_df, x=plot_df.index, y='Close', color='#00E676', linewidth=2.5, ax=ax)
    ax.axhspan(result["gz_lower"], result["gz_upper"], color='#388E3C', alpha=0.35, label='Altın Bölge (0.5 - 0.618)')
    ax.axhline(result["tp"], color='#FFD600', linestyle='--', linewidth=1.5, label='Kâr Al (1.618)')
    ax.axhline(result["stop_loss"], color='#D32F2F', linestyle='-.', linewidth=2, label='Stop Loss (Ana Dip)')
    ax.axhline(result["last_high"], color='#9E9E9E', linestyle=':', linewidth=1, alpha=0.5, label='Son Zirve (Fib 0)')
    
    ax.set_title(f"{ticker} | Durum: {result['signal_type']}", color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Zaman (Son Barlar)", color='lightgray')
    ax.set_ylabel("Fiyat", color='lightgray')
    ax.legend(loc='upper left', facecolor='#1E1E1E', labelcolor='white', framealpha=0.9, edgecolor='#333333')
    
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    ax.tick_params(colors='lightgray')
    for spine in ax.spines.values():
        spine.set_color('#2A2A2A')
        
    plt.tight_layout()
    return fig

# ==========================================
# 5. STREAMLIT ARAYÜZ MİMARİSİ
# ==========================================
st.title("🦅 HexaTrades Golden Zone Kuantitatif Tarayıcı")
st.markdown("Piyasadaki hacimli hisseler taranır. Sistem, **Altın Bölge'de olanları** listeler.")
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Tarama Kriterleri")
    market_selection = st.selectbox("Piyasa Seçimi:", list(MARKET_CONFIGS.keys()))
    timeframe_selection = st.selectbox("Periyot:", list(TIMEFRAME_MAP.keys()))
    scan_button = st.button("🚀 Tarayıcıyı Başlat", use_container_width=True)

with col2:
    if scan_button:
        interval = TIMEFRAME_MAP[timeframe_selection]
        mkt_config = MARKET_CONFIGS[market_selection]
        
        st.info("TradingView veritabanından en hacimli hisseler çekiliyor...")
        tv_symbols = get_all_market_symbols(mkt_config)
            
        if not tv_symbols:
            st.error("Bağlantı hatası: TradingView'den hisse listesi alınamadı.")
            st.stop()
            
        st.success(f"Başarılı! Toplam **{len(tv_symbols)}** hisse analiz ediliyor. İşlem sürüyor...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        signals = []
        
        completed = 0
        total_tickers = len(tv_symbols)
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_ticker = {executor.submit(analyze_ticker, t, mkt_config, interval): t for t in tv_symbols}
            
            for future in as_completed(future_to_ticker):
                res = future.result()
                if res:
                    signals.append(res)
                    
                completed += 1
                if completed % 5 == 0 or completed == total_tickers:
                    progress_bar.progress(completed / total_tickers)
                    status_text.text(f"İşlenen: {completed} / {total_tickers} | Bulunan Fırsat: {len(signals)}")
                
        progress_bar.empty()
        status_text.empty()
        
        st.markdown("---")
        if not signals:
            st.warning("Seçilen periyotta Altın Bölge kriterlerini karşılayan hisse bulunamadı. Lütfen zaman periyodunu değiştirerek tekrar deneyin.")
        else:
            signals.sort(key=lambda x: "Pusu" in x["signal_type"])
            
            st.success(f"🎯 Toplam **{len(signals)}** adet hissede Golden Zone fırsatı tespit edildi!")
            
            for res in signals:
                ticker = res["ticker"]
                current_p = float(res["current_price"].iloc[0]) if isinstance(res["current_price"], pd.Series) else float(res["current_price"])
                tp_p = float(res["tp"])
                kar_potansiyeli = ((tp_p - current_p) / current_p) * 100
                
                is_pusu = "Pusu" in res['signal_type']
                icon = "⏳" if is_pusu else "🟢"
                
                tv_link = f"https://www.tradingview.com/chart/?symbol={mkt_config['tv_prefix']}{ticker}"
                
                with st.expander(f"{icon} {ticker} | Mevcut: {current_p:.2f} | Kâr Al: {tp_p:.2f} (+%{kar_potansiyeli:.2f})", expanded=False):
                    col_info, col_chart = st.columns([1, 4])
                    
                    with col_info:
                        st.markdown(f"**Durum:** {res['signal_type']}")
                        st.markdown(f"**Giriş Bandı:** {res['gz_lower']:.2f} - {res['gz_upper']:.2f}")
                        st.markdown(f"**Stop (Dip):** {res['stop_loss']:.2f}")
                        st.markdown(f"[📊 TradingView'da Aç]({tv_link})")
                        
                    with col_chart:
                        fig = plot_setup(res)
                        st.pyplot(fig)
                        plt.close(fig)
