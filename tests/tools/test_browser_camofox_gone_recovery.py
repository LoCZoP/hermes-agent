"""Regression tests for the Camofox "wedged on a stale tab id" fix.

The camofox server recycles idle tabs/sessions and can restart its browser child
between calls. After that, the wrapper's cached tab id is dead and the server
answers **410 Gone** (tab_destroyed / page_crashed / tab_timeout, all marked
retryable: create_new_tab) or **404** for a tab the current browser never knew.

The original `_navigate_tab` only recovered on 404, so a 410 re-raised and the
session stayed wedged; the other tab actions had no recovery at all. These tests
pin the new behavior: 404 AND 410 are both treated as "tab gone" and recovered by
recreating the tab and retrying.
"""

import json
from unittest.mock import MagicMock, patch

from tools.browser_camofox import (
    _navigate_tab,
    _with_tab,
    _request,
    camofox_navigate,
    camofox_snapshot,
    _CamofoxGoneError,
)


def _resp(status, json_data=None):
    import requests as _req
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data if json_data is not None else {}
    r.text = json.dumps(json_data or {})
    r.content = b""
    if status >= 400:
        def _raise():
            raise _req.HTTPError(f"HTTP {status}", response=r)
        r.raise_for_status = _raise
    else:
        r.raise_for_status = MagicMock()
    return r


# ---------------------------------------------------------------------------
# _request: gone-status mapping
# ---------------------------------------------------------------------------

class TestRequestGoneMapping:
    @patch("tools.browser_camofox.requests.get")
    def test_410_raises_gone_error(self, mock_get, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_get.return_value = _resp(410, {"error": "Tab was destroyed. Open a new tab.",
                                           "code": "tab_destroyed", "retryable": True,
                                           "recovery": "create_new_tab"})
        try:
            _request("get", "/tabs/dead/snapshot")
            assert False, "expected _CamofoxGoneError"
        except _CamofoxGoneError as e:
            assert e.status_code == 410
            assert e.payload.get("recovery") == "create_new_tab"

    @patch("tools.browser_camofox.requests.get")
    def test_404_raises_gone_error(self, mock_get, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_get.return_value = _resp(404, {"error": "Tab not found"})
        try:
            _request("get", "/tabs/never-known/snapshot")
            assert False, "expected _CamofoxGoneError"
        except _CamofoxGoneError as e:
            assert e.status_code == 404

    @patch("tools.browser_camofox.requests.get")
    def test_500_is_not_a_gone_error(self, mock_get, monkeypatch):
        # 5xx is a server-side transient, handled by backoff-retry, NOT tab recreation.
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        import requests as _req
        mock_get.return_value = _resp(500, {"error": "boom"})
        try:
            _request("get", "/tabs/x/snapshot")
            assert False, "expected HTTPError"
        except _CamofoxGoneError:
            assert False, "500 must NOT map to a gone error"
        except _req.HTTPError:
            pass  # correct: raised-for-status path


# ---------------------------------------------------------------------------
# _navigate_tab: recovers on BOTH 404 and 410 (the reported wedge)
# ---------------------------------------------------------------------------

class TestNavigateTabRecovery:
    def _seed_session(self, task_id, tab_id, url="http://localhost:5173"):
        from tools.browser_camofox import _sessions, _sessions_lock
        with _sessions_lock:
            _sessions[task_id] = {"user_id": "u1", "tab_id": tab_id,
                                  "session_key": "task_x", "last_url": url,
                                  "managed": True, "adopt_existing_tab": False}

    def _cleanup(self, task_id):
        from tools.browser_camofox import _sessions, _sessions_lock
        with _sessions_lock:
            _sessions.pop(task_id, None)

    @patch("tools.browser_camofox.load_config")
    @patch("tools.browser_camofox.requests.post")
    def test_navigate_recovers_on_410(self, mock_post, mock_cfg, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_cfg.return_value = {"browser": {"camofox": {}}}
        self._seed_session("t410", "dead-tab-id")
        # 1st POST: navigate to the stale tab -> 410 gone; 2nd POST: recreate -> new tab
        mock_post.side_effect = [
            _resp(410, {"error": "Tab was destroyed. Open a new tab.",
                        "code": "tab_destroyed", "recovery": "create_new_tab"}),
            _resp(200, {"tabId": "fresh-tab-id", "url": "http://localhost:5173/"}),
        ]
        session, data = _navigate_tab("t410", "http://localhost:5173")
        assert session["tab_id"] == "fresh-tab-id", "session must be re-pointed to the new tab"
        assert mock_post.call_count == 2
        self._cleanup("t410")

    @patch("tools.browser_camofox.load_config")
    @patch("tools.browser_camofox.requests.post")
    def test_navigate_recovers_on_404(self, mock_post, mock_cfg, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_cfg.return_value = {"browser": {"camofox": {}}}
        self._seed_session("t404", "dead-tab-id")
        mock_post.side_effect = [
            _resp(404, {"error": "Tab not found"}),
            _resp(200, {"tabId": "fresh-404", "url": "http://localhost:5173/"}),
        ]
        session, data = _navigate_tab("t404", "http://localhost:5173")
        assert session["tab_id"] == "fresh-404"
        self._cleanup("t404")


# ---------------------------------------------------------------------------
# _with_tab: other actions recreate-and-retry after a 410
# ---------------------------------------------------------------------------

class TestWithTabRecovery:
    def _seed_session(self, task_id, tab_id, url="http://localhost:5173"):
        from tools.browser_camofox import _sessions, _sessions_lock
        with _sessions_lock:
            _sessions[task_id] = {"user_id": "u1", "tab_id": tab_id,
                                  "session_key": "task_x", "last_url": url,
                                  "managed": True, "adopt_existing_tab": False}

    def _cleanup(self, task_id):
        from tools.browser_camofox import _sessions, _sessions_lock
        with _sessions_lock:
            _sessions.pop(task_id, None)

    @patch("tools.browser_camofox.load_config")
    @patch("tools.browser_camofox.requests.post")
    @patch("tools.browser_camofox.requests.get")
    def test_snapshot_recovers_on_410_and_succeeds(self, mock_get, mock_post, mock_cfg, monkeypatch):
        """The QA wedge: a snapshot (or any tab action) after an idle GC used to fail
        forever; now it recreates the tab and retries the same action."""
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_cfg.return_value = {"browser": {"camofox": {}}}
        self._seed_session("trec", "dead-tab")
        # snapshot GET: 1st -> 410 gone, 2nd (retry after recreate) -> 200 with snapshot
        mock_get.side_effect = [
            _resp(410, {"error": "Tab was destroyed. Open a new tab.", "recovery": "create_new_tab"}),
            _resp(200, {"snapshot": "- heading \"hi\"", "refsCount": 1}),
        ]
        # recreate POST /tabs -> 200 new tab
        mock_post.return_value = _resp(200, {"tabId": "reborn-tab", "url": "http://localhost:5173/"})

        result = json.loads(camofox_snapshot(task_id="trec"))
        assert result["success"] is True
        assert result["element_count"] == 1
        # the session must now point at the recreated tab, not the dead one
        from tools.browser_camofox import _sessions
        assert _sessions["trec"]["tab_id"] == "reborn-tab"
        self._cleanup("trec")
