import logging

from flask import Flask, request, jsonify, render_template
from rag_pipeline import build_rag_and_answer

app = Flask(__name__)
app.logger.setLevel(logging.INFO)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json(silent=True) or {}
    user_query = data.get('query')
    if not user_query:
        return jsonify({'error': 'Missing "query" parameter'}), 400
    try:
        answer = build_rag_and_answer(user_query)
        return jsonify({'answer': answer})
    except SystemExit as exc:
        app.logger.info("Request aborted: %s", exc)
        raise
    except Exception as exc:
        app.logger.exception("Unhandled exception during request")
        return jsonify({'error': str(exc)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
