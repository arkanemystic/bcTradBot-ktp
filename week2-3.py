# --- Algorithmic Trading Bot Development: Session Deliverable ---
# This single, runnable script covers all aspects of the hands-on lab:
# 1. Logs historical price data for BTC and ETH to a CSV file.
# 2. Loads that data and calculates key technical indicators (SMAs, RSI, Bollinger Bands).
# 3. Generates and prints simple "BUY" or "SELL" signals based on SMA crossover logic.
# 4. Visualizes the price and indicators on a chart.

import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import argparse

# --- Configuration ---
# Data Storage (as discussed in Slide 4)
LOG_FILE = 'crypto_price_history.csv'
SYMBOLS = ['BTC', 'ETH']

# API Configuration
API_URL = 'https://min-api.cryptocompare.com/data/pricemulti'
API_PARAMS = {'fsyms': ','.join(SYMBOLS), 'tsyms': 'USD', 'api_key': os.getenv('CRYPTOCOMPARE_API_KEY')}

# Technical Indicator Settings (from Slides 7, 8, 9)
SMA_SHORT_WINDOW = 10  # Short-term moving average period
SMA_LONG_WINDOW = 30   # Long-term moving average period
BBANDS_WINDOW = 20     # Bollinger Bands moving average period
RSI_WINDOW = 14        # RSI calculation period

# --- Part 1: Data Logging and Management ---

def fetch_prices():
    """Fetches the current prices for the configured symbols."""
    try:
        response = requests.get(API_URL, params=API_PARAMS)
        response.raise_for_status()
        data = response.json()
        prices = {symbol: data[symbol]['USD'] for symbol in SYMBOLS}
        return prices
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
    except KeyError:
        print(f"Error: Unexpected API response format. Response: {response.text}")
    return None

def log_data_continuously():
    """
    Logs the current prices to a CSV file every minute.
    This function runs in an infinite loop. Press Ctrl+C to stop.
    """
    print(f"Starting price logging to '{LOG_FILE}'... Press Ctrl+C to stop.")
    while True:
        prices = fetch_prices()
        if prices:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            price_data = {'timestamp': timestamp, **{f"{s}_price": p for s, p in prices.items()}}

            df_new = pd.DataFrame([price_data])
            header = not os.path.exists(LOG_FILE)
            df_new.to_csv(LOG_FILE, mode='a', header=header, index=False)
            
            log_message = ', '.join([f'{s}: ${p:,.2f}' for s, p in prices.items()])
            print(f"[{timestamp}] Logged: {log_message}")

        time.sleep(1) # Wait for 60 seconds

# --- Part 2: Trading Logic and Indicator Calculation ---

def calculate_technical_indicators(df, symbol='BTC'):
    """
    Calculates SMAs, Bollinger Bands, and RSI for the price data.
    """
    price_col = f"{symbol}_price"
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")

    # 1. Simple Moving Averages (SMA) - Slide 7
    df[f'SMA_{SMA_SHORT_WINDOW}'] = df[price_col].rolling(window=SMA_SHORT_WINDOW).mean()
    df[f'SMA_{SMA_LONG_WINDOW}'] = df[price_col].rolling(window=SMA_LONG_WINDOW).mean()

    # 2. Bollinger Bands - Slide 9
    df['BB_MIDDLE'] = df[price_col].rolling(window=BBANDS_WINDOW).mean()
    std_dev = df[price_col].rolling(window=BBANDS_WINDOW).std()
    df['BB_UPPER'] = df['BB_MIDDLE'] + (std_dev * 2)
    df['BB_LOWER'] = df['BB_MIDDLE'] - (std_dev * 2)

    # 3. Relative Strength Index (RSI) - Slide 8
    delta = df[price_col].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_WINDOW, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_WINDOW, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def generate_trading_signals(df):
    """
    Generates 'Buy' and 'Sell' signals based on the SMA Crossover strategy.
    """
    df['Signal'] = 0  # 1 for Buy, -1 for Sell
    
    # SMA Crossover Logic (as described in Slide 7)
    # A "Buy" signal is when the short-term SMA crosses *above* the long-term SMA.
    buy_condition = (df[f'SMA_{SMA_SHORT_WINDOW}'] > df[f'SMA_{SMA_LONG_WINDOW}']) & \
                    (df[f'SMA_{SMA_SHORT_WINDOW}'].shift(1) <= df[f'SMA_{SMA_LONG_WINDOW}'].shift(1))
    
    # A "Sell" signal is when the short-term SMA crosses *below* the long-term SMA.
    sell_condition = (df[f'SMA_{SMA_SHORT_WINDOW}'] < df[f'SMA_{SMA_LONG_WINDOW}']) & \
                     (df[f'SMA_{SMA_SHORT_WINDOW}'].shift(1) >= df[f'SMA_{SMA_LONG_WINDOW}'].shift(1))

    df.loc[buy_condition, 'Signal'] = 1
    df.loc[sell_condition, 'Signal'] = -1
    
    signals_df = df[df['Signal'] != 0].copy()
    signals_df['Signal_Text'] = signals_df['Signal'].apply(lambda x: 'BUY' if x == 1 else 'SELL')
    
    return df, signals_df

# --- Part 3: Visualization ---

def plot_analysis(df, signals_df, symbol='BTC'):
    """
    Plots the price, indicators, and trading signals on a chart.
    """
    price_col = f"{symbol}_price"
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # --- Top Plot: Price, Indicators, and Signals ---
    ax1.plot(df.index, df[price_col], label=f'{symbol} Price', color='lightgray', alpha=0.9)
    ax1.plot(df.index, df[f'SMA_{SMA_SHORT_WINDOW}'], label=f'{SMA_SHORT_WINDOW}-Period SMA', color='orange', linestyle='--')
    ax1.plot(df.index, df[f'SMA_{SMA_LONG_WINDOW}'], label=f'{SMA_LONG_WINDOW}-Period SMA', color='purple', linestyle='--')
    ax1.fill_between(df.index, df['BB_LOWER'], df['BB_UPPER'], color='skyblue', alpha=0.2, label='Bollinger Bands')

    ax1.scatter(signals_df[signals_df['Signal'] == 1].index, 
                signals_df[signals_df['Signal'] == 1][price_col],
                label='Buy Signal', marker='^', color='green', s=150, zorder=5)

    ax1.scatter(signals_df[signals_df['Signal'] == -1].index, 
                signals_df[signals_df['Signal'] == -1][price_col],
                label='Sell Signal', marker='v', color='red', s=150, zorder=5)

    ax1.set_title(f'{symbol}/USD Price Analysis with Trading Signals', fontsize=16)
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)

    # --- Bottom Plot: RSI ---
    ax2.plot(df.index, df['RSI'], label='RSI', color='dodgerblue')
    ax2.axhline(70, linestyle='--', color='red', alpha=0.7, label='Overbought (70)')
    ax2.axhline(30, linestyle='--', color='green', alpha=0.7, label='Oversold (30)')
    
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylabel('RSI')
    ax2.set_xlabel('Date')
    ax2.set_ylim(0, 100)
    ax2.legend()
    
    plt.tight_layout()
    print("\nDisplaying analysis chart... Close the chart window to exit.")
    plt.show()

# --- Main Execution Block ---
def run_analysis(symbol='BTC'):
    """Loads data, runs calculations, and shows the plot."""
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file '{LOG_FILE}' not found. Run the logger first with 'log' command.")
        return

    print(f"Loading data from '{LOG_FILE}'...")
    df = pd.read_csv(LOG_FILE, parse_dates=['timestamp'], index_col='timestamp')

    # Ensure there's enough data for the longest indicator window
    if len(df) < SMA_LONG_WINDOW:
        print(f"Not enough data to generate indicators. Need at least {SMA_LONG_WINDOW} data points, but found {len(df)}.")
        print("Please let the logger run for a longer period.")
        return
        
    print("Calculating technical indicators...")
    df = calculate_technical_indicators(df, symbol=symbol)

    print("Generating trading signals based on SMA Crossover...")
    df_with_signals, signals = generate_trading_signals(df)

    print("\n--- Trading Signals Generated ---")
    if not signals.empty:
        print(signals[[f'{symbol}_price', 'Signal_Text']].to_string())
    else:
        print("No new trading signals were generated in the current dataset.")
    print("---------------------------------")
    
    plot_analysis(df_with_signals, signals, symbol=symbol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Algorithmic Trading Data Logger and Analyzer.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Sub-parser for the "log" command
    log_parser = subparsers.add_parser(
        'log', 
        help='Start the data logger to continuously fetch and save price data.'
    )

    # Sub-parser for the "analyze" command
    analyze_parser = subparsers.add_parser(
        'analyze', 
        help='Analyze the logged data, generate signals, and display a chart.'
    )
    analyze_parser.add_argument(
        '--symbol', 
        type=str, 
        default='BTC', 
        choices=SYMBOLS,
        help='The crypto symbol to analyze (e.g., BTC, ETH).'
    )

    args = parser.parse_args()

    if args.command == 'log':
        try:
            log_data_continuously()
        except KeyboardInterrupt:
            print("\nData logging stopped by user.")
    elif args.command == 'analyze':
        run_analysis(symbol=args.symbol)
