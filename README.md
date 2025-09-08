# Keep the Future Human Survival RPG

This project is an experimental role‑playing game inspired by the themes of the
[Keep the Future Human essay contest](https://keepthefuturehuman.ai/contest/).
It began as a minimal example of how to call Google's Gemini API with the
`google-generativeai` library and has grown into a playable RPG. The repository
now includes:

- **CLI game (`cli_game.py`)** – play through the terminal.
- **Flask web service (`web_service.py`)** – interact with the same mechanics via a browser.
- **Character definitions (`characters/`)** – each folder describes a playable
  character whose actions are generated with the model.
- **Gemini API wrapper (`main.py`)** – a simple script demonstrating direct text
  generation calls.

Together these pieces form a small game exploring how humans might survive and
thrive in future scenarios.

## Setup

1. Copy `.env.template` to `.env` and set `GEMINI_API_KEY` to your Gemini API key. The script loads this file automatically.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Demo

Run the RPG demo interactively:

```bash
python cli_game.py
```

### Web Service

Start a small Flask web service exposing the same game logic:

```bash
python web_service.py
```
Then open `http://127.0.0.1:7860/` in a browser. The page presents
character choices as radio buttons. Select a character and submit to see
possible actions, then choose an action and press **Send** to view the
character's response.

### Logging

Both entry points respect the `LOG_LEVEL` environment variable to control
verbosity. Set `LOG_LEVEL=DEBUG` for more detailed output. The default
level is `INFO`.

## License

This project is licensed under the terms of the [GNU General Public License
v3.0 or later](LICENSE).

