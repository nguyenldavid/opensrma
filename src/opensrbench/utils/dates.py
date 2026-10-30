"""Lenient date parsing for bibliographic metadata and free text."""

import calendar
import datetime
import re

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}  # yapf: disable

_ISO = re.compile(r"^(\d{4})(?:[-/](\d{1,2}))?(?:[-/](\d{1,2}))?")
_MONTH_WORD = r"(jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)[a-z]*\.?"
_DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH_WORD},?\s+(\d{{4}})\b", re.I
)
_MONTH_DAY_YEAR = re.compile(
    rf"\b{_MONTH_WORD}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I
)
_MONTH_YEAR = re.compile(rf"\b{_MONTH_WORD},?\s+(\d{{4}})\b", re.I)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _safe_date(year: int, month: int, day: int) -> datetime.date | None:
    """Builds a date, clamping month to [1, 12] and day to the month length."""
    if not 1000 <= year <= 9999:
        return None
    month = min(max(month, 1), 12)
    day = min(max(day, 1), calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)


def parse_date(
    value: object, end_of_period: bool = False
) -> datetime.date | None:
    """Parses ISO-like or English free-text dates.

    Args:
      value: A string such as "2024-03-01", "2024/3", "March 2024",
        "1 March 2024", "March 1, 2024" or "2024".
      end_of_period: When only a month or year is known, return the last day
        of that period (useful for search cut-offs) instead of the first.

    Returns:
      A date or None.
    """
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    m = _ISO.match(text)
    if m:
        year = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else (12 if end_of_period else 1)
        if m.group(3):
            day = int(m.group(3))
        else:
            day = 31 if end_of_period else 1
        return _safe_date(year, month, day)
    m = _DAY_MONTH_YEAR.search(text)
    if m:
        return _safe_date(
            int(m.group(3)), _MONTHS[m.group(2)[:3].lower()], int(m.group(1))
        )
    m = _MONTH_DAY_YEAR.search(text)
    if m:
        return _safe_date(
            int(m.group(3)), _MONTHS[m.group(1)[:3].lower()], int(m.group(2))
        )
    m = _MONTH_YEAR.search(text)
    if m:
        return _safe_date(
            int(m.group(2)),
            _MONTHS[m.group(1)[:3].lower()],
            31 if end_of_period else 1,
        )
    m = _YEAR.search(text)
    if m:
        year = int(m.group(1))
        return (
            datetime.date(year, 12, 31)
            if end_of_period
            else datetime.date(year, 1, 1)
        )
    return None


def to_iso(value: datetime.date | None) -> str | None:
    """Formats a date as YYYY-MM-DD (or None)."""
    return value.isoformat() if value else None


def to_pubmed(value: datetime.date) -> str:
    """Formats a date as PubMed's YYYY/MM/DD."""
    return value.strftime("%Y/%m/%d")
