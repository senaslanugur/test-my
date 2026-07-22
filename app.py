import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Kuantitatif Trader Terminali", page_icon="📈", layout="wide"
)


# 1. Veri İndirme ve Teknik İndikatör Motoru
@st.cache_data(ttl=3600)
def load_market_data(ticker, period="1y", interval="1d"):
  try:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df
  except Exception as e:
    st.error(f"Veri çekme hatası ({ticker}): {e}")
    return pd.DataFrame()


def calculate_indicators(df, atr_period=14, mult=2.0, rsi_period=14):
  # True Range & ATR Hesaplama
  df["H-L"] = df["High"] - df["Low"]
  df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
  df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
  df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
  df["ATR"] = df["TR"].rolling(window=atr_period).mean()

  # RSI Hesaplama
  delta = df["Close"].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
  rs = gain / loss
  df["RSI"] = 100 - (100 / (1 + rs))

  # ATR Trailing Stop (Long Yönlü Basitleştirilmiş Mantık)
  df["TrailingStop"] = np.nan
  trailing_stops = []
  curr_ts = 0.0

  for i in range(len(df)):
    close = df["Close"].iloc.iloc[i] if hasattr(df["Close"].iloc, 'iloc') else df["Close"].iloc[i] # Güvenli erişim
    # Alternatif standart erişim:
    pass

  # Vektörel/İteratif ATR Stop
  close_arr = df["Close"].values
  atr_arr = df["ATR"].values
  ts_arr = np.zeros(len(df))

  for i in range(len(df)):
    if np.isnan(atr_arr[i]):
      continue
    basic_ts = close_arr[i] - (mult * atr_arr[i])
    if i == 0:
      ts_arr[i] = basic_ts
    else:
      prev_ts = ts_arr[i - 1]
      prev_close = close_arr[i - 1]
      if close_arr[i] > prev_ts and prev_close > prev_ts:
        ts_arr[i] = max(prev_ts, basic_ts)
      else:
        ts_arr[i] = basic_ts

  df["TrailingStop"] = ts_arr
  return df


# 2. State Machine & Sinyal Üretimi
def generate_signals(df):
  signals = []
  state = "OUT"  # "OUT" veya "IN" (Long)

  entry_prices = []
  exit_prices = []
  trade_states = []

  for i in range(len(df)):
    close = df["Close"].iloc[i]
    ts = df["TrailingStop"].iloc[i]
    rsi = df["RSI"].iloc[i]

    if state == "OUT":
      # Taze Alım Koşulu: Trend onayı ve Aşırı satım/dönüş filtresi
      if not np.isnan(ts) and close > ts and rsi < 70:
        state = "IN"
        signals.append("BUY")
        entry_prices.append(close)
        exit_prices.append(np.nan)
        trade_states.append("ENTRY")
      else:
        signals.append("HOLD")
        entry_prices.append(np.nan)
        exit_prices.append(np.nan)
        trade_states.append("OUT")

    elif state == "IN":
      # Stop Seviyesi Altına Sarkma Durumu (Çıkış)
      if close < ts:
        state = "OUT"
        signals.append("SELL")
        entry_prices.append(np.nan)
        exit_prices.append(close)
        trade_states.append("EXIT")
      else:
        signals.append("HOLD")
        entry_prices.append(np.nan)
        exit_prices.append(np.nan)
        trade_states.append("IN")

  df["Signal"] = signals
  df["TradeState"] = trade_states
  return df


# 3. Streamlit Arayüz Tasarımı
st.title("🛡️ Kuantitatif ATR & State Machine Terminali")
st.sidebar.header("Parametre Paneli")

ticker_input = st.sidebar.text_input("Hisse / Varlık Kodu", value="THYAO.IS")
period_input = st.sidebar.selectbox(
    "Veri Periyodu", ["6mo", "1y", "2y", "5y"], index=1
)
atr_p = st.sidebar.slider("ATR Periyodu", 5, 30, 14)
atr_m = st.sidebar.slider("ATR Çarpanı (Multiplier)", 1.0, 5.0, 2.0, 0.5)
rsi_p = st.sidebar.slider("RSI Periyodu", 5, 25, 14)

# Veriyi Yükle ve Hesapla
raw_df = load_market_data(ticker_input, period=period_input)

if not raw_df.empty:
  processed_df = calculate_indicators(
      raw_df, atr_period=atr_p, mult=atr_m, rsi_period=rsi_p
  )
  final_df = generate_signals(processed_df)

  # Metrik Özetleri
  col1, col2, col3, col4 = st.columns(4)
  last_close = final_df["Close"].iloc[-1]
  prev_close = final_df["Close"].iloc[-2]
  daily_change = ((last_close - prev_close) / prev_close) * 100
  last_rsi = final_df["RSI"].iloc[-1]
  last_ts = final_df["TrailingStop"].iloc[-1]
  current_state = final_df["TradeState"].iloc[-1]

  col1.metric("Son Fiyat", f"{last_close:.2f} TL", f"{daily_change:.2f}%")
  col2.metric("Güncel RSI", f"{last_rsi:.1f}")
  col3.metric("ATR Trailing Stop", f"{last_ts:.2f} TL")
  col4.metric(
      "Sistem Durumu (State)",
      current_state,
      delta="Pozisyonda" if current_state == "IN" else "Nakit/Dışarıda",
      delta_color="normal" if current_state == "IN" else "off",
  )

  # Grafik Sekmeleri
  tab1, tab2, tab3 = st.tabs(
      ["📊 Fiyat & Trailing Stop", "📋 Sinyal Tablosu", "🔬 Backtest Özeti"]
  )

  with tab1:
    st.subheader(f"{ticker_input} - Teknik Görünüm")
    chart_data = final_df[["Close", "TrailingStop"]].copy()
    st.line_chart(chart_data)

  with tab2:
    st.subheader("Son İşlem Sinyalleri ve Durum Logları")
    action_rows = final_df[final_df["Signal"].isin(["BUY", "SELL"])].tail(10)
    st.dataframe(action_rows[["Close", "ATR", "RSI", "Signal", "TrailingStop"]])

  with tab3:
    st.subheader("Strateji Performans Simülasyonu")
    # Basit Getiri Hesaplama
    final_df["Market_Return"] = final_df["Close"].pct_change().fillna(0)
    final_df["Strategy_Position"] = (
        final_df["TradeState"].shift(1).fillna("OUT") == "IN"
    ).astype(int)
    final_df["Strategy_Return"] = (
        final_df["Market_Return"] * final_df["Strategy_Position"]
    )

    cum_market = (1 + final_df["Market_Return"]).cumprod() - 1
    cum_strategy = (1 + final_df["Strategy_Return"]).cumprod() - 1

    perf_df = pd.DataFrame(
        {
            "Al-Unut (Buy & Hold)": cum_market * 100,
            "ATR Stratejisi": cum_strategy * 100,
        }
    )
    st.line_chart(perf_df)

    total_trades = len(final_df[final_df["Signal"] == "BUY"])
    st.info(f"Test Edilen Dönem İçindeki Toplam İşlem (Alım) Sayısı: {total_trades}")
else:
  st.warning("Seçilen kriterlere uygun veri bulunamadı veya yüklenemedi.")
