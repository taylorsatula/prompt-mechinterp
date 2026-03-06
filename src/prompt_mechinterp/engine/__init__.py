"""MI engine — core attention capture and logit lens analysis.

SELF-CONTAINMENT DESIGN:
    run_analysis.py is designed to be scp'd to a remote GPU box and executed
    with zero local imports. All model auto-discovery logic from model_adapter.py
    is inlined in run_analysis.py (clearly marked). When developing locally, you
    can import from model_adapter directly.

    This dual-existence means changes to model discovery logic must be synced
    between model_adapter.py (clean importable version) and the inlined copy
    in run_analysis.py (self-contained deployment version).
"""
