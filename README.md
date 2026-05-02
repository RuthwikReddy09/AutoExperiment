# AutoExperiment Production-Ready Demo

A LangGraph-based autonomous ML experimentation agent using:

- PyTorch + torchvision ResNet-18 on CIFAR-10
- LangGraph multi-agent workflow
- LangChain + OpenAI LLM planner
- Optuna hyperparameter optimization
- MLflow experiment tracking
- YAML configuration
- Heuristic fallback when no OpenAI API key is available

## Folder structure

```text
autoexperiment/
  graph.py
  state.py
  agents/
    observer.py
    planner.py
    executor.py
    evaluator.py
  training/
    train.py
    model.py
    data.py
  tracking/
    mlflow_utils.py
  configs/
    base_config.yaml
  main.py
```

## Quick start

```bash
unzip autoexperiment_prod.zip
cd autoexperiment_prod
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Run with LLM planner

```bash
export OPENAI_API_KEY="your_api_key_here"
python main.py --config configs/base_config.yaml --use-llm-planner
```

Or set:

```yaml
agent:
  use_llm_planner: true
```

## View MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

## Notes

- Default config uses CIFAR-10 with ResNet-18.
- `quick_debug: true` trains on a subset so you can test the full pipeline quickly.
- Set `quick_debug: false` for full CIFAR-10 training.
