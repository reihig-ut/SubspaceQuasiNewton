fmt:
	black .
	isort . --profile black

unzip:
	if [ -e results.zip ]; then \
		rm -rf results/; \
		unzip results.zip; \
		rm results.zip; \
	fi
	if [ -e mlruns.zip ]; then \
		rm -rf mlruns/; \
		unzip mlruns.zip; \
		rm mlruns.zip; \
	fi

zip:
	zip -r results.zip results/
	zip -r mlruns.zip mlruns/

run:
	for f in configs/unexecuted/*.json; do \
		python main.py $$f && mv $$f configs/executed/; \
	done
