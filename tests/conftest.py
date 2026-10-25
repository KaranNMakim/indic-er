"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.models import InputRecord  # noqa: E402


@pytest.fixture
def jayesh_records() -> list[InputRecord]:
    """The worked example from PLAN.md: three records, one person.

    Used across the pipeline tests because it exercises every hard thing at
    once — Gujarati script, a fused honorific, an abbreviated address, a
    landmark synonym, a city alias and a missing phone.
    """
    return [
        InputRecord(
            id="r1",
            name="Jayeshbhai Patel",
            address="12, Shreeji Soc., Nr. Swaminarayan Mandir, Maninagar, Ahmedabad 380008",
            phone="+91 98250 12345",
            source="kyc",
        ),
        InputRecord(
            id="r2",
            name="જયેશ પટેલ",
            address="12 Shriji Society, opp Swaminarayan Temple, Maninagar, Ahmedabad",
            phone="9825012345",
            source="wallet",
        ),
        InputRecord(
            id="r3",
            name="J. Patil",
            address="Shreeji Society, Maninagar, Amdavad, 380 008",
            source="logistics",
        ),
    ]


@pytest.fixture
def unrelated_record() -> InputRecord:
    """A different Patel, in a different city, with a different phone."""
    return InputRecord(
        id="x1",
        name="Nilesh Patel",
        address="88, Gokul Apt., Sector 12, Gandhinagar 382012",
        phone="9900011122",
        source="kyc",
    )
