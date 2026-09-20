"""Read-only view of the gpushare GPU service, proxied for the dashboard."""

from .client import GpushareClient

__all__ = ["GpushareClient"]
