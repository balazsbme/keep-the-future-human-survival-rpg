from flask import Flask, request, jsonify
from rag_pipeline import build_rag_and_answer

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json(silent=True) or {}
    user_query = data.get('query')
    if not user_query:
        return jsonify({'error': 'Missing "query" parameter'}), 400
    try:
        answer = build_rag_and_answer(user_query)
        return jsonify({'answer': answer})
    except Exception as exc:
        # In production you might want to log the exception.
        return jsonify({'error': str(exc)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
