import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import warnings

# Yfinance hatalarını terminalde gizle
warnings.filterwarnings("ignore")

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
        "range": [0, 400] # API ban yememek için en hacimli 400 hisse
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
    İstatistiksel Swing High/Low noktalarını vektörel olarak bulan 
    ve Golden Zone'u (Yaklaşanlar dahil) hesaplayan mimari.
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
    
    pullback_bars = df.iloc[last_pivot[0] + 1 : ]
    if pullback_bars.empty: 
        return None
        
    touched_zone = pullback_bars['Low'].min() <= gz_upper
    
    curr_close = closes[-1]
    curr_low = lows[-1]
    curr_open = opens[-1]
    
    signal = False
    signal_type = ""
    
    # Yaklaşma Mesafesi Hesaplama (Golden Zone'un yüzde kaç üzerinde?)
    distance_to_zone_pct = (curr_close - gz_upper) / gz_upper
    
    # SİNYAL 1: Kusursuz Pine Script Ret İşlemi
    if curr_low <= gz_upper and curr_close > gz_upper and curr_close > curr_open:
        signal = True
        signal_type = "🎯 Kusursuz Ret (Tam Alım)"
        
    # SİNYAL 2: Radar/Pusu Modu (İçeride bekliyor)
    elif gz_lower <= curr_close <= gz_upper:
        signal = True
        signal_type = "⏳ Pusu Modu: Altın Bölge İçinde"
        
    # SİNYAL 3: Güvenli Alım (Daha önce değdi, şimdi yukarı gidiyor)
    elif touched_zone and curr_close > gz_upper and curr_close > curr_open:
        signal = True
        signal_type = "🚀 Bölgeden Onaylı Çıkış"
        
    # SİNYAL 4: YAKLAŞANLAR (Bölgeye %2.5 veya daha az kalmış)
    elif 0 < distance_to_zone_pct <= 0.025:
        signal = True
        signal_type = f"👀 Yaklaşıyor (Mesafe: %{distance_to_zone_pct*100:.1f})"
        
    if signal:
        tp_1618 = leg_high + (rng * 0.618)
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
st.markdown("Piyasadaki hacimli hisseler toplu (bulk) olarak çekilir ve hata payı olmadan analiz edilir.")
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
            
        yf_tickers = [f"{t.replace('.', '-')}{mkt_config['yf_suffix']}" for t in tv_symbols]
        
        st.info(f"Yfinance üzerinden {len(yf_tickers)} hissenin verisi IP engeline takılmamak için toplu indiriliyor. Bu işlem 10-20 saniye sürebilir...")
        
        period = "1mo" if interval == "1h" else ("1y" if interval == "1d" else "2y")
        
        # TOPLU İNDİRME MİMARİSİ (YF Rate Limit Aşımı)
        try:
            df_all = yf.download(yf_tickers, period=period, interval=interval, group_by='ticker', threads=True, progress=False, show_errors=False)
        except Exception as e:
            st.error("Veri indirme sırasında hata oluştu. Lütfen tekrar deneyin.")
            st.stop()
            
        signals = []
        
        st.info("Veriler indirildi, Golden Zone algoritmaları çalıştırılıyor...")
        progress_bar = st.progress(0)
        
        for i, (symbol, yf_ticker) in enumerate(zip(tv_symbols, yf_tickers)):
            try:
                # MultiIndex sütun mimarisine göre veriyi ayır
                if len(tv_symbols) > 1:
                    df = df_all[yf_ticker].dropna(how='all')
                else:
                    df = df_all.dropna(how='all')
                    
                if not df.empty and len(df) > 50:
                    df.columns = [str(c).capitalize() for c in df.columns]
                    result = find_golden_zone_vectorized(df)
                    if result and result["signal"]:
                        result["ticker"] = symbol
                        result["df"] = df
                        result["current_price"] = df['Close'].iloc[-1]
                        signals.append(result)
            except Exception:
                pass
            
            progress_bar.progress((i + 1) / len(tv_symbols))
                
        progress_bar.empty()
        
        st.markdown("---")
        if not signals:
            st.warning("Seçilen periyotta Altın Bölge kriterlerini veya yaklaşma (radar) şartını karşılayan hisse bulunamadı.")
        else:
            # Önce tam alım noktaları, sonra yaklaşanlar görünecek şekilde sırala
            signals.sort(key=lambda x: "Yaklaşıyor" in x["signal_type"])
            
            st.success(f"🎯 Toplam **{len(signals)}** adet hissede Golden Zone fırsatı veya radar sinyali tespit edildi!")
            
            for res in signals:
                ticker = res["ticker"]
                current_p = float(res["current_price"].iloc[0]) if isinstance(res["current_price"], pd.Series) else float(res["current_price"])
                tp_p = float(res["tp"])
                kar_potansiyeli = ((tp_p - current_p) / current_p) * 100
                
                is_yaklasan = "Yaklaşıyor" in res['signal_type']
                icon = "👀" if is_yaklasan else "🎯"
                
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
