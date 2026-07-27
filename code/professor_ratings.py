"""
professor_ratings.py

Loads RateMyProfessor ratings scraped for SFU professors
(data/sfu_rmp/sfu_professors.csv) and matches them to instructor names coming
from the course schedule data, so the scheduler can prefer sections taught by
higher-rated professors.
"""

import os
import re
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RMP_CSV = os.path.join(_SCRIPT_DIR, "..", "data", "sfu_rmp", "sfu_professors.csv")


def _normalize_name(name):
    """Lowercase and strip punctuation/extra whitespace so e.g. 'Toby J. Donaldson'
    and 'Toby Donaldson' are comparable."""
    if not isinstance(name, str) or not name:
        return ""
    name = re.sub(r'[^a-zA-Z\s]', ' ', name)
    return ' '.join(name.lower().split())


def load_ratings(csv_path=DEFAULT_RMP_CSV):
    """
    Returns a dict mapping normalized professor name -> average rating.
    A name appearing on multiple rows (e.g. cross-listed departments) is
    averaged across those rows.
    """
    df = pd.read_csv(csv_path)
    totals = {}
    counts = {}
    for _, row in df.iterrows():
        key = _normalize_name(row.get('name', ''))
        rating = row.get('rating')
        if not key or pd.isna(rating):
            continue
        totals[key] = totals.get(key, 0.0) + float(rating)
        counts[key] = counts.get(key, 0) + 1
    return {name: totals[name] / counts[name] for name in totals}


def make_rating_lookup(csv_path=DEFAULT_RMP_CSV):
    """
    Returns a `get_rating(name) -> float | None` function backed by the RMP
    dataset, for scoring section instructors during schedule generation.
    """
    ratings = load_ratings(csv_path)

    def get_rating(name):
        return ratings.get(_normalize_name(name))

    return get_rating
