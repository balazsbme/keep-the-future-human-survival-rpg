import json
import sys
import os
from unittest.mock import patch

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import rag_service


def test_index_page_served():
    with rag_service.app.test_client() as client:
        response = client.get('/')
        assert response.status_code == 200
        assert b'id="send"' in response.data


def test_query_endpoint_returns_answer():
    with patch('rag_service.build_rag_and_answer', return_value='test-answer'):
        with rag_service.app.test_client() as client:
            response = client.post('/query', json={'query': 'hi'})
            assert response.status_code == 200
            assert response.get_json() == {'answer': 'test-answer'}


def test_query_endpoint_requires_query():
    with rag_service.app.test_client() as client:
        response = client.post('/query', json={})
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
