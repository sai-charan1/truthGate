.PHONY: setup ingest eval serve test clean

setup: ingest

ingest:
	python -c "from dotenv import load_dotenv; load_dotenv()"
	python src/ingestion/scraper.py
	python src/ingestion/chunker.py
	python src/ingestion/indexer.py
	@echo "✅ Ingestion complete"

eval:
	python run_eval.py

eval-verbose:
	python run_eval.py --verbose

serve:
	python app.py

ask:
	python cli.py ask "$(Q)"

clean:
	rm -rf data/raw data/chunks data/index evals/eval_results.json

docker-up:
	docker compose up --build

test:
	python -c "from src.qa_engine import query; r = query('How do you declare a path parameter in FastAPI?'); print('answer_type:', r['answer_type']); assert r['answer_type'] == 'answered', f'Expected answered, got {r[\"answer_type\"]}'; print('✅ Smoke test passed')"
