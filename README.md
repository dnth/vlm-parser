# vlm-parser
Extract visual information from your images using VLM.

## Installation
Install [`uv`](https://docs.astral.sh/uv/).

Next, create a virtual environment and install the dependencies.

```bash
uv venv
```

Activate the virtual environment.

```bash
source .venv/bin/activate
```

Then install the dependencies.

```bash
uv sync
```

Download the spacy English model.

```bash
python -m spacy download en_core_web_sm
```

## Usage
Run the Gradio app.

```bash
python scripts/app.py
```


## Results

![Results](./assets/result.png)
![Results](./assets/result_1.png)
