"""Pont de controle vers le moteur de synthese."""

from .osc import JamBridge, NullSink, OscSink, RecordingSink, create_sink

__all__ = ["JamBridge", "NullSink", "OscSink", "RecordingSink", "create_sink"]
