PY ?= python

install:      ## create venv deps
	$(PY) -m pip install -r requirements-dev.txt
train:        ## generate data, backtest, train, write reports
	$(PY) scripts/train.py
llm:          ## compare Claude vs GPT vs Gemini
	$(PY) scripts/compare_llms.py
test:
	$(PY) -m pytest -q
api:
	uvicorn api.index:app --reload --port 8000
ui:
	streamlit run app/streamlit_app.py
all: train test llm
