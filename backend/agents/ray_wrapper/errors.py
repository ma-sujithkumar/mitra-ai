"""Structured errors raised by the Epic-3 Ray execution layer."""

from __future__ import annotations


class RayWrapperError(RuntimeError):
    """Base exception for Ray-wrapper failures."""


class RayInitializationError(RayWrapperError):
    """Raised when neither an external nor a local Ray runtime can start."""


class RaySubmissionError(RayWrapperError):
    """Raised when a validated training job cannot be submitted."""


class RayExecutionError(RayWrapperError):
    """Raised for invalid or inconsistent results returned by a Ray worker."""
