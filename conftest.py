"""Test setup: lets pytest import the project modules from the repo root.

If flask_cors is not installed, a stand-in is used. The tests do not use CORS, and
requirements.txt installs the real package on a normal setup.
"""
import sys
import types

try:
    import flask_cors  # noqa: F401
except ImportError:
    stub = types.ModuleType("flask_cors")
    stub.CORS = lambda app, **kwargs: app
    sys.modules["flask_cors"] = stub
