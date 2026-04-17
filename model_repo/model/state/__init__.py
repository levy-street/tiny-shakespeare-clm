"""Model state package.

Split `schema.py` into further submodules as the state grows — e.g.
`linguistic.py`, `flow.py`, `speaker.py`, `verse.py` — as long as the public
`ModelState` remains a single Pydantic class re-exported from here.
"""

from .schema import ModelState

__all__ = ["ModelState"]
