# 1. Import the ccxt library
import ccxt

# 2. Create an "instance" of the Coinbase exchange
exchange = ccxt.coinbase()

# 3. Define the trading symbols we want data for
#    (Note: a "ticker" symbol is just a pair like BTC/USD)
btc_symbol = 'BTC/USD'
eth_symbol = 'ETH/USD'

# 4. Use the fetch_ticker() function to get the latest market data
btc_ticker = exchange.fetch_ticker(btc_symbol)
eth_ticker = exchange.fetch_ticker(eth_symbol)

# 5. The data is a dictionary. The price is stored in the 'last' key.
btc_price = btc_ticker['last']
eth_price = eth_ticker['last']

# 6. Print the results in a readable format!
print(f"The current price of {btc_symbol} on Coinbase is ${btc_price}")
print(f"The current price of {eth_symbol} on Coinbase is ${eth_price}")
