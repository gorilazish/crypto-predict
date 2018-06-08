FROM = '01/01/2014'
TO = '01/05/2018'

all: ohlcv

clean:
	rm -f data/raw/*.csv


twitter:
	python src/features/aggregate_tweets.py

ohlcv:
	python src/data/get_btc_prices.py $(FROM) $(TO)
	python src/data/add_header.py
	python src/data/timestamp_to_datetime.py
	python src/features/aggregate_prices.py
	twitter
	tree data
