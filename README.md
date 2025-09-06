# Google Cloud Text Generation Example

This repository contains a tiny Python script demonstrating how to call Google Cloud's text generation API using the `google-genai` library.

## Setup

1. Copy `.env.template` to `.env` and update the values for your project. The script loads this file automatically.
2. Configure Google Cloud authentication (for example with `gcloud auth application-default login`).
3. Install dependencies:
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
