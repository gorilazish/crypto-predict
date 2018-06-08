FROM = '01/01/2014'
TO = '01/05/2018'

all: ohlcv

clean:
	rm -f data/raw/*.csv

ohlcv:
	python src/data/get_btc_prices.py $(FROM) $(TO)
	# python src/data/add_header.py
	# python src/data/timestamp_to_datetime.py
	tree data