FROM = '01/01/2014'

all: ohlcv

clean:
	rm -f data/raw/*.csv

ohlcv:
	python src/data/get_btc_prices.py $(FROM)
	tree data