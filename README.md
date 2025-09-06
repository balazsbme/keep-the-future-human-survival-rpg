# Google Generative AI Text Generation Example

This repository contains a tiny Python script demonstrating how to call Google's Gemini API using the `google-generativeai` library.

## Setup

1. Copy `.env.template` to `.env` and set `GEMINI_API_KEY` to your Gemini API key. The script loads this file automatically.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script and provide a prompt when prompted:

```bash
python main.py
```

The script sends your prompt to the `gemini-1.5-flash` model and prints the generated response.

## License

MIT

