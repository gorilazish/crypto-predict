FROM = '19/06/2018'

all: twitter ohlcv merge graphs

clean:
	#rm -f data/interim/*.csv
	#rm -f data/external/*.csv
	#rm -f data/processed/*.csv
	#rm -f data/raw/*.csv

twitter:
	# python src/data/fetch_live_tweets.py
	# RUN ONLY ONCE
	# takes around 30mins
	python3 src/data/prep_tweets.py
	# takes around 5mins
	# python3 src/features/prepare_cols.py

	# RUN FOR EVERY TIMEFRAME
	# python3 src/features/aggregate_tweets.py
	tree data

ohlcv:
	# RUN ONLY ONCE
	python src/data/get_btc_prices.py $(FROM)
	python src/data/add_header.py
	python src/data/timestamp_to_datetime.py

	# RUN FOR EVERY TIMEFRAME
	python src/features/aggregate_prices.py
	tree data

merge:
	python src/features/merge.py

graphs:
	# python src/visualization/describe_data.py
	python src/visualization/price_tweet_volume.py
	python src/visualization/explore.py

lstm:
	python src/models/lstm.py

lr:
	python src/models/linear_regression.py

mlp:
	python src/models/mlp.py
