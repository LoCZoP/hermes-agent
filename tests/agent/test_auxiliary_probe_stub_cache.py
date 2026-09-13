"""Regression tests for the aux-vision ``_AuxProbeClientStub`` cache-poisoning bug.

Bug (reproduced on this host, Sep 2026): ``check_vision_requirements`` resolves the
configured aux-vision provider inside ``aux_probe_mode()`` so tool-gating is cheap.
That resolution flows through ``_get_cached_client``, whose *build path* wrote the
returned client into ``_client_cache`` **directly** — bypassing the ``_store_cached_client``
stub-guard. So a probe-mode ``_AuxProbeClientStub`` got cached, and the first *real*
vision call in the process that resolved the same cache key was served the stub and
crashed:

    _AuxProbeClientStub used as a real client (attribute 'chat')

The fix makes ``_get_cached_client`` never cache (and never return) a probe stub: a
stub is an availability signal, not a client. These tests pin that contract.
"""

import pytest

from agent import auxiliary_client as aux


@pytest.fixture(autouse=True)
def _clean_aux_cache():
    """Isolate the process-global client cache around each test."""
    with aux._client_cache_lock:
        aux._client_cache.clear()
    yield
    with aux._client_cache_lock:
        aux._client_cache.clear()


class _RealClient:
    """Stand-in for a genuine OpenAI/SDK client: has a ``.chat`` surface."""
    def __init__(self):
        self.chat = _Chat()

    def close(self):
        pass


class _Chat:
    def completions(self):  # pragma: no cover - only attribute existence is asserted
        return object()


def _fake_resolve_provider_client(provider, model=None, async_mode=False,
                                  explicit_base_url=None, explicit_api_key=None,
                                  api_mode=None, main_runtime=None, is_vision=False,
                                  task=None, **_):
    """Mimic the real router: in probe mode the builders return a stub, otherwise a
    real client. This is exactly the shape that let the stub leak into the cache."""
    if aux._aux_probe_active():
        return aux._AuxProbeClientStub(api_key="k", base_url="https://example.invalid"), model
    return _RealClient(), model or "default-model"


@pytest.fixture
def probe_or_real_builder(monkeypatch):
    monkeypatch.setattr(aux, "resolve_provider_client", _fake_resolve_provider_client)


class TestNoStubCachePoisoning:
    def test_probe_mode_get_does_not_cache_or_return_a_stub(self, monkeypatch, probe_or_real_builder):
        monkeypatch.setattr(aux, "_peek_pool_entry", lambda _p: None)
        with aux.aux_probe_mode():
            client, model = aux._get_cached_client("opencode-go", async_mode=False, is_vision=True)
        # The guard returns (None, model) for a stub — never a stub, never cached.
        assert client is None
        with aux._client_cache_lock:
            stubs = [c for (_k, (c, _d, _l)) in aux._client_cache.items()
                     if isinstance(c, aux._AuxProbeClientStub)]
        assert stubs == [], "a probe stub must never be written to _client_cache"

    def test_real_call_after_probe_gets_a_real_client(self, monkeypatch, probe_or_real_builder):
        """The exact reported sequence: a startup probe, then a real vision call in the
        same process. Before the fix the real call was served the cached stub."""
        monkeypatch.setattr(aux, "_peek_pool_entry", lambda _p: None)
        with aux.aux_probe_mode():
            aux._get_cached_client("opencode-go", async_mode=False, is_vision=True)
        # Real (non-probe) resolution must now build and return a genuine client.
        client, model = aux._get_cached_client("opencode-go", async_mode=False, is_vision=True)
        assert client is not None
        assert not isinstance(client, aux._AuxProbeClientStub)
        # And the very thing that used to crash: attribute access for the chat surface.
        assert client.chat is not None

    def test_store_cached_client_still_rejects_stubs(self):
        """Existing guard remains intact (defense in depth)."""
        key = ("p", False, "", None, "", (), False, "", "", None, "m")
        aux._store_cached_client(key, aux._AuxProbeClientStub(), "m")
        with aux._client_cache_lock:
            assert key not in aux._client_cache
