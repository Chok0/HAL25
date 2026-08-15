"""halj — jam generatif local : la guitare pilote un drone accorde et un rythme.

L'analyse (Python) et la synthese (SuperCollider) sont deux process distincts
relies par OSC : voir `halj.bridge.osc` pour le protocole.
"""

from .config import Config
from .notes import Key

__version__ = "0.1.0"
__all__ = ["Config", "Key", "__version__"]
