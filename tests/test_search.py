import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


MOCK_ROWS = [
    ("C:/Projects/app/auth.py",   "lines 10-45",  "def login(user, password):\n    ...", 0.12),
    ("C:/Projects/app/utils.py",  "lines 1-30",   "def read_file(path):\n    ...",       0.35),
    ("C:/Projects/api/server.go", "lines 80-120", "func handleAuth(w, r) {\n}",          0.58),
]

def _make_pool(rows=None):
    rows = rows if rows is not None else MOCK_ROWS
    cur = MagicMock()
    cur.fetchall.return_value = rows
    cur.__enter__ = lambda s: s
    cur.__exit__ = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cur
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, cur

def _mock_encode(model_mock, vector=None):
    vec = vector or [0.1] * 384
    model_mock.encode.return_value.tolist.return_value = vec


class TestSearch:

    @patch("search._model")
    @patch("search._table_name", return_value="seekindex_code_embeddings")
    def test_returns_correct_number_of_results(self, _tbl, mock_model):
        _mock_encode(mock_model)
        from search import search
        pool, _ = _make_pool()
        assert len(search(pool, "login function")) == len(MOCK_ROWS)

    @patch("search._model")
    @patch("search._table_name", return_value="seekindex_code_embeddings")
    def test_score_is_one_minus_distance(self, _tbl, mock_model):
        _mock_encode(mock_model)
        from search import search
        pool, _ = _make_pool(rows=[("f.py", "L1", "code", 0.25)])
        assert search(pool, "query")[0]["score"] == pytest.approx(0.75, abs=1e-4)

    @patch("search._model")
    @patch("search._table_name", return_value="seekindex_code_embeddings")
    def test_empty_query_returns_empty_list(self, _tbl, mock_model):
        from search import search
        pool, cur = _make_pool()
        assert search(pool, "   ") == []
        cur.execute.assert_not_called()

    @patch("search._model")
    @patch("search._table_name", return_value="seekindex_code_embeddings")
    def test_result_fields_present(self, _tbl, mock_model):
        _mock_encode(mock_model)
        from search import search
        pool, _ = _make_pool(rows=[("a.py", "L5", "x = 1", 0.0)])
        r = search(pool, "variable")[0]
        assert {"filename", "location", "code", "score"} <= r.keys()

    @patch("search._model")
    @patch("search._table_name", return_value="seekindex_code_embeddings")
    def test_ext_filter_adds_where_clause(self, _tbl, mock_model):
        _mock_encode(mock_model)
        from search import search
        pool, cur = _make_pool()
        search(pool, "anything", ext_filter=["py", "go"])
        (sql, _), _ = cur.execute.call_args
        assert "~" in sql

    @patch("search._model")
    @patch("search._table_name", return_value="seekindex_code_embeddings")
    def test_path_filter_adds_ilike_clause(self, _tbl, mock_model):
        _mock_encode(mock_model)
        from search import search
        pool, cur = _make_pool()
        search(pool, "anything", path_filter="src/")
        (sql, _), _ = cur.execute.call_args
        assert "ILIKE" in sql.upper()

    @patch("search._model")
    @patch("search._table_name", return_value="seekindex_code_embeddings")
    def test_top_k_passed_as_param(self, _tbl, mock_model):
        _mock_encode(mock_model)
        from search import search
        pool, cur = _make_pool()
        search(pool, "query", top_k=42)
        (_, params), _ = cur.execute.call_args
        assert 42 in params


class TestTableName:

    @patch("search.cocoindex.utils.get_target_storage_default_name", return_value="seekindex_code_embeddings")
    def test_valid_name_passes(self, _mock):
        from search import _table_name
        assert _table_name() == "seekindex_code_embeddings"

    @patch("search.cocoindex.utils.get_target_storage_default_name", return_value="bad name; DROP TABLE users--")
    def test_invalid_name_raises(self, _mock):
        from search import _table_name
        with pytest.raises(ValueError, match="Unexpected table name"):
            _table_name()


class TestPrintResults:

    def _capture(self, fn, *args, **kwargs) -> str:
        buf = StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            fn(*args, **kwargs)
        finally:
            sys.stdout = old
        return buf.getvalue()

    def test_json_output_is_valid(self):
        from search import _print_results
        results = [{"filename": "a.py", "location": "L1", "code": "x=1", "score": 0.9}]
        parsed = json.loads(self._capture(_print_results, results, as_json=True))
        assert parsed[0]["score"] == 0.9

    def test_text_output_contains_filename(self):
        from search import _print_results
        results = [{"filename": "my_file.py", "location": "L1", "code": "pass", "score": 0.8}]
        assert "my_file.py" in self._capture(_print_results, results, as_json=False)

    def test_snippet_truncated_at_300_chars(self):
        from search import _print_results
        results = [{"filename": "f.py", "location": "L1", "code": "x" * 500, "score": 0.5}]
        assert "x" * 301 not in self._capture(_print_results, results, as_json=False)


class TestDatabaseUrl:

    def test_raises_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("COCOINDEX_DATABASE_URL", raising=False)
        from config import database_url
        with pytest.raises(EnvironmentError, match="COCOINDEX_DATABASE_URL"):
            database_url()

    def test_returns_url_when_set(self, monkeypatch):
        monkeypatch.setenv("COCOINDEX_DATABASE_URL", "postgresql://localhost/test")
        from config import database_url
        assert database_url() == "postgresql://localhost/test"
