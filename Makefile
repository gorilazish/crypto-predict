FROM = '01/01/2018'

all: ohlcv

clean:
	#rm -f data/interim/*.csv

twitter:
	# takes around 30mins
	python3 src/data/prep_tweets.py
	# takes around 5mins
	#python3 src/features/prepare_cols.py
	#python3 src/features/aggregate_tweets2.py

ohlcv:
	#python src/data/get_btc_prices.py $(FROM)
	python src/data/add_header.py
	python src/data/timestamp_to_datetime.py
	python src/features/aggregate_prices.py
	#tree data

graphs:
	python src/visualization/describe_data.py
	python src/visualization/neg_pos_distribution.py
	python src/visualization/price_timeframe.py
	python src/visualization/relations.py
	python src/visualization/tweets_volume.py
