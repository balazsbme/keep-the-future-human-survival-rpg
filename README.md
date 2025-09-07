# Google Generative AI Text Generation Example

This repository contains a tiny Python script demonstrating how to call Google's Gemini API using the `google-generativeai` library.
It also includes a small role-playing game (RPG) demo that can be played either through the command line or via a web service.

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
python example_game.py
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

## License

This project is licensed under the terms of the [GNU General Public License
v3.0 or later](LICENSE).

