"""Decimal CAD amounts shared by submission and usage APIs."""

from decimal import Decimal
from typing import Annotated

from pydantic import Field

from .protocol import Identifier, Model, Submission

Money = Annotated[Decimal, Field(ge=0, le=1_000_000_000, max_digits=16, decimal_places=6)]


class UsageCap(Model):
    # Required: explicit null removes a cap; an omitted value is an error.
    cap: Money | None


class RunLookup(Model):
    job_id: Identifier


class SetRunCap(RunLookup, UsageCap):
    pass


class MeteredSubmission(Submission):
    usage_cap: Money | None = None
