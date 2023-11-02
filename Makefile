reformat:
	python -m black .
	python -m isort .

bench_mark: reformat
	python utils/bench_mark.py