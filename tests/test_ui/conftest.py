"""Shared fixtures for UI tests."""

from unittest.mock import MagicMock


class FakeSessionState:
    """Minimal st.session_state substitute.

    Supports attribute access, item access, .get(), __contains__, and __delitem__
    — mirroring Streamlit's SessionState API without starting a runtime.
    """

    def __init__(self, **kwargs):
        object.__setattr__(self, "_store", dict(kwargs))

    def __getattr__(self, name):
        store = object.__getattribute__(self, "_store")
        try:
            return store[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        object.__getattribute__(self, "_store")[name] = value

    def __contains__(self, name):
        return name in object.__getattribute__(self, "_store")

    def __getitem__(self, name):
        return object.__getattribute__(self, "_store")[name]

    def __setitem__(self, name, value):
        object.__getattribute__(self, "_store")[name] = value

    def __delitem__(self, name):
        del object.__getattribute__(self, "_store")[name]

    def get(self, name, default=None):
        return object.__getattribute__(self, "_store").get(name, default)


def make_st(**session_kwargs) -> MagicMock:
    """Return a mock ``st`` module with a pre-populated FakeSessionState."""
    st = MagicMock()
    st.session_state = FakeSessionState(**session_kwargs)
    return st
