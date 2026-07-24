#!/usr/bin/env python3
# decisionTreeModel.py - loads all data, trains the fullness-prediction decision tree, 
# and saves it to a bundle for later use by decisionTree.py.

# Usage:
#    python decisionTreeModel.py [flags]

# Flags:
#    --data-dir <path>            data/ folder to read from (default ../data)
#    --model-path <path>          where to save the trained bundle (default ./decisionTree_model.joblib)
#    --test-sizes <list>          comma-seperate list of sizes to compare, model picks best
#    --lookback-terms <int>       how many prior terms to average the same-week historic fill rate over (default 4)
#    --skip-instructor-api        skip the SFU course-outlines API lookup (faster, works offline)
#    --max-instructor-lookups <int>
#                                 cap on unique sections queried against the
#                                  SFU API while building the training set (default 300)

# Example:
#    python decisionTreeModel.py --test-sizes 0.1,0.2,0.3 --max-instructor-lookups 500


# Required: pip install pandas openpyxl scikit-learn joblib requests
import sys
import os
import re
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import openpyxl
import requests
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decisionTree_model.joblib")

REG_OPEN = {
    # opens in the prior term
    "spring": (11, 13),
    "summer": (3, 3),
    "fall": (7, 6),
}
TERM_DIGIT = {"spring": 1, "summer": 4, "fall": 7}


# 1. Parse enrollment date -> term, week of registration, term code
def parse_enrollment_date(enrollment_date_str):
    date_obj = datetime.strptime(enrollment_date_str, "%Y-%m-%d")
    y = date_obj.year
    spring_open = datetime(y - 1, *REG_OPEN["spring"])
    summer_open = datetime(y, *REG_OPEN["summer"])
    fall_open = datetime(y, *REG_OPEN["fall"])
    next_spring_open = datetime(y, *REG_OPEN["spring"])
    if date_obj < summer_open:
        term_name, term_year, reg_open = "spring", y, spring_open
    elif date_obj < fall_open:
        term_name, term_year, reg_open = "summer", y, summer_open
    elif date_obj < next_spring_open:
        term_name, term_year, reg_open = "fall", y, fall_open
    else:
        term_name, term_year, reg_open = "spring", y + 1, next_spring_open
    #Determine week of registration 
    week_of_registration = max(1, (date_obj - reg_open).days // 7 + 1)
    term_code = (term_year - 1900) * 10 + TERM_DIGIT[term_name]
    return term_name, term_year, week_of_registration, term_code, reg_open

# Determine when registrain opens
def _week_of_registration_for_date(date_obj, term_name, term_year):
    if term_name == "spring":
        reg_open = datetime(term_year - 1, *REG_OPEN["spring"])
    elif term_name == "summer":
        reg_open = datetime(term_year, *REG_OPEN["summer"])
    else:
        reg_open = datetime(term_year, *REG_OPEN["fall"])
    return max(1, (date_obj - reg_open).days // 7 + 1)

# Dervive the season/year from XXXX
def term_name_from_code(term_code):
    digit = term_code % 10
    year = 1900 + term_code // 10
    name = {1: "spring", 4: "summer", 7: "fall"}.get(digit, "unknown")
    return name, year

# Put codes into standard form
def normalize_code(subject, catnbr):
    return f"{str(subject).strip().upper()}{str(catnbr).strip().upper()}"

# Used for safe comparison, ex. CMPT 225 vs cmpt225 vs CMPT-225
def normalize_token(token):
    return re.sub(r"[\s\-]", "", str(token)).strip().upper()

# 2. Load a term's course-offering file (either report format) into a normalized long-format DataFrame.
def _load_long_format(ws, term_code):
    # Parses the 'database' sheet layout: one row per (date, section)
    header_row = None
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=15, values_only=True), start=1):
        if row and row[0] and str(row[0]).strip().casefold() == "date":
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()
    cols = [c for c in ws.iter_rows(min_row=header_row, max_row=header_row, values_only=True)][0]
    # Normalize column names by stripping spaces and converting to lowercase for safe lookup
    col_idx = {str(name).strip().casefold(): i for i, name in enumerate(cols) if name is not None}
    def get_val(row_data, *possible_names, default=None):
        # Safely fetches a value from the row using multiple possible column header names.
        for name in possible_names:
            idx = col_idx.get(str(name).strip().casefold())
            if idx is not None and idx < len(row_data):
                return row_data[idx]
        return default
    records = []
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        date_val = get_val(row, "Date")
        if not isinstance(date_val, datetime):
            continue
        subject = get_val(row, "Subject", "Subj")
        catnbr = get_val(row, "CatNbr", "Cat Nbr", "Catalog Number", "CatalogNbr")
        if not subject or not catnbr:
            continue
        term_name, term_year = term_name_from_code(term_code)
        records.append({
            "term_code": term_code,
            "date": date_val,
            "week_of_reg": _week_of_registration_for_date(date_val, term_name, term_year),
            "subject": subject,
            "catnbr": catnbr,
            "section": get_val(row, "Sect", "Section", "Sec", "Class Sect"),
            "course_title": get_val(row, "Course Title", "Title", "Description"),
            "max_enrol": get_val(row, "MaxEnrol", "Max Enrol", "Capacity", "Cap"),
            "act_enrol": get_val(row, "ActEnrol", "Act Enrol", "Enrolled", "Enrollment"),
            "percent_filled": get_val(row, "%Filled", "% Filled", "Percent Filled"),
            "last_sem_taught_5yrs": get_val(row, "LastSemTaught5Yrs", "Last Sem Taught 5Yrs"),
            "sem_taught_2yrs": get_val(row, "#SemTaught2Yrs", "SemTaught2Yrs", "# Sem Taught 2Yrs"),
        })
    return pd.DataFrame.from_records(records)

def _load_pivot_format(ws, term_code):
    # Parses the 'pivot.percent.filled' sheet layout: one row per section, %Filled spread across date columns, with forward-filled course info.
    header_row = None
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=20, values_only=True), start=1):
        if row and row[0] and str(row[0]).strip().casefold() == "subject":
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()
    week_label_row = [r for r in ws.iter_rows(min_row=header_row - 2, max_row=header_row - 2, values_only=True)][0]
    header = [r for r in ws.iter_rows(min_row=header_row, max_row=header_row, values_only=True)][0]
    # Case-insensitive lookup so a slightly renamed header (e.g. 'Section' instead of 'Sect') doesn't crash the loader — just falls back to None.
    static_cols = {str(name).strip().casefold(): i for i, name in enumerate(header[:9]) if name is not None}
    def col(row_data, *possible_names):
        for name in possible_names:
            idx = static_cols.get(name.strip().casefold())
            if idx is not None and idx < len(row_data):
                return row_data[idx]
        return None
    date_col_idx = [i for i, v in enumerate(header) if isinstance(v, datetime)]
    # forward-fill sparse week-group labels across the date columns
    week_label = {}
    last_label = None
    for i in date_col_idx:
        if week_label_row[i]:
            last_label = week_label_row[i]
        week_label[i] = last_label
    # Helper to extract week number
    def label_to_week_num(label):
        if not label:
            return None
        m = re.search(r"Week (\d+)", label)
        if not m:
            return None
        n = int(m.group(1))
        return n + 10 if "Classes" in label else n
    term_name, term_year = term_name_from_code(term_code)
    records = []
    last_subject = last_catnbr = last_title = last_5yr = last_2yr = None
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        if col(row, "Subject"):
            last_subject = col(row, "Subject")
        if col(row, "CatNbr", "Cat Nbr"):
            last_catnbr = col(row, "CatNbr", "Cat Nbr")
        if col(row, "Course Title", "Title"):
            last_title = col(row, "Course Title", "Title")
        if col(row, "LastSemTaught5Yrs"):
            last_5yr = col(row, "LastSemTaught5Yrs")
        if col(row, "#SemTaught2Yrs", "SemTaught2Yrs"):
            last_2yr = col(row, "#SemTaught2Yrs", "SemTaught2Yrs")
        section = col(row, "Sect", "Section", "Sec")
        max_enrol = col(row, "MaxEnrol", "Max Enrol", "Capacity", "Cap")
        if not last_subject or not last_catnbr or not section:
            continue
        for i in date_col_idx:
            val = row[i] if i < len(row) else None
            if val is None:
                continue
            week_num = label_to_week_num(week_label.get(i))
            records.append({
                "term_code": term_code,
                "date": header[i],
                "week_of_reg": week_num if week_num is not None else
                               _week_of_registration_for_date(header[i], term_name, term_year),
                "subject": last_subject,
                "catnbr": last_catnbr,
                "section": section,
                "course_title": last_title,
                "max_enrol": max_enrol,
                "act_enrol": np.nan,
                "percent_filled": val,
                "last_sem_taught_5yrs": last_5yr,
                "sem_taught_2yrs": last_2yr,
            })
    return pd.DataFrame.from_records(records)

 # Handles both raw fractions (0.53) and percent-strings ('53%')
def _parse_percent(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    if s.endswith("%"):
        s = s[:-1]
        try:
            return float(s) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan
_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Converts a string like "04-Jul" or "4 Jul" into a (day, month) pair
def _parse_day_month(s):
    m = re.match(r"\s*(\d{1,2})[\s\-]?([A-Za-z]{3})", str(s))
    if not m:
        return None
    month = _MONTH_ABBR.get(m.group(2).lower())
    if month is None:
        return None
    return int(m.group(1)), month

# Loads a CSV file in the long-format layout (one row per date/section) into a DataFrame.
def _load_csv_long_format(path, term_code):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    term_name, term_year = term_name_from_code(term_code)
    def parse_date(s):
        dm = _parse_day_month(s)
        if dm is None:
            return pd.NaT
        day, month = dm
        # Spring registration opens in November of the PRIOR calendar year,
        # so Nov/Dec dates in a spring-term file belong to term_year - 1.
        year = term_year - 1 if (term_name == "spring" and month in (11, 12)) else term_year
        try:
            return datetime(year, month, day)
        except ValueError:
            return pd.NaT
    df["_date"] = df["Date"].apply(parse_date)
    df = df.dropna(subset=["_date"])
    out = pd.DataFrame({
        "term_code": term_code,
        "date": df["_date"],
        "week_of_reg": df["_date"].apply(lambda d: _week_of_registration_for_date(d, term_name, term_year)),
        "subject": df.get("Subject"),
        "catnbr": df.get("CatNbr"),
        "section": df.get("Sect"),
        "course_title": df.get("Course Title"),
        "max_enrol": pd.to_numeric(df.get("MaxEnrol"), errors="coerce"),
        "act_enrol": pd.to_numeric(df.get("ActEnrol"), errors="coerce"),
        "percent_filled": df.get("%Filled").apply(_parse_percent) if "%Filled" in df.columns else np.nan,
        "last_sem_taught_5yrs": df.get("LastSemTaught5Yrs"),
        "sem_taught_2yrs": pd.to_numeric(df.get("#SemTaught2Yrs"), errors="coerce"),
    })
    return out.dropna(subset=["subject", "catnbr", "section"])

# Loads a course-offering file (either report format) into a normalized long-format DataFrame.
def load_offering_file(path, term_code):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _load_csv_long_format(path, term_code)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if "database" in wb.sheetnames:
        return _load_long_format(wb["database"], term_code)
    return _load_pivot_format(wb[wb.sheetnames[0]], term_code)

# Extracts a 4-digit term code from the filename (e.g. "1214" for Summer 2021).
def _extract_term_code(fname):
    matches = re.findall(r"(?<!\d)(1\d{3})(?!\d)", fname)
    return int(matches[0]) if matches else None

# Loads every .xlsx/.xls/.csv file (any name containing a 4-digit term code) under data/<subfolder>/ into one normalized long DataFrame.
def _load_offerings_folder(data_dir, subfolder):
    folder = os.path.join(data_dir, subfolder)
    frames = []
    if not os.path.isdir(folder):
        return pd.DataFrame()
    for fname in os.listdir(folder):
        #ignore the prerequisites file and the database.xsls files file
        if fname.startswith("sfu_prerequisites"):
            continue
        if fname.startswith("database") and fname.lower().endswith(".xlsx"):
            continue
        #ignore any file that doesn't have a valid extension
        if not fname.lower().endswith((".xlsx", ".xls", ".csv")):
            continue
        term_code = _extract_term_code(fname)
        if term_code is None:
            print(f"Warning: Skipping {fname}: couldn't find a 4-digit term code in the filename.")
            continue
        df = load_offering_file(os.path.join(folder, fname), term_code)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    # Concatenate all Dataframes into one
    all_df = pd.concat(frames, ignore_index=True)
    all_df["percent_filled"] = pd.to_numeric(all_df["percent_filled"], errors="coerce")
    all_df["max_enrol"] = pd.to_numeric(all_df["max_enrol"], errors="coerce")
    all_df["sem_taught_2yrs"] = pd.to_numeric(all_df["sem_taught_2yrs"], errors="coerce")
    all_df = all_df.dropna(subset=["percent_filled"])
    return all_df

# Loads all course-offering files in the data/courseOfferings/ folder into a single DataFrame.
def load_all_offerings(data_dir=DATA_DIR):
    return _load_offerings_folder(data_dir, "courseOfferings")

# Loads the data/courseFillByWeek/percent-filled-<termCode>.xlsx reports
def load_course_fill_by_week(data_dir=DATA_DIR):
    return _load_offerings_folder(data_dir, "courseFillByWeek")

# Pre-aggregates courseFillByWeek data into two dicts so historic_fill_rate lookups are O(1)
def _build_fill_week_index(fill_df):
    course_idx, section_idx = {}, {}
    if fill_df.empty:
        return course_idx, section_idx
    course_g = fill_df.groupby(["subject", "catnbr", "week_of_reg", "term_code"])["percent_filled"].mean()
    for (subj, cat, wk, term), val in course_g.items():
        course_idx.setdefault((str(subj).upper(), str(cat).upper(), wk), {})[term] = val
    section_g = fill_df.groupby(["subject", "catnbr", "section", "week_of_reg", "term_code"])["percent_filled"].mean()
    for (subj, cat, sec, wk, term), val in section_g.items():
        section_idx.setdefault((str(subj).upper(), str(cat).upper(), str(sec).upper(), wk), {})[term] = val
    return course_idx, section_idx

# Find average %Filled for this course at Same week of registration for up to X lookback terms
def get_historic_fill_rate(course_idx, section_idx, subject, catnbr, section, week_of_reg,
                            current_term_code, lookback_terms=4):
    subject, catnbr, section = str(subject).upper(), str(catnbr).upper(), str(section).upper()
    sec_terms = section_idx.get((subject, catnbr, section, week_of_reg), {})
    course_terms = course_idx.get((subject, catnbr, week_of_reg), {})
    #Falls back to any course term if section data is not available
    prior_terms = sorted((t for t in set(sec_terms) | set(course_terms) if t < current_term_code), reverse=True)
    rates = []
    for t in prior_terms[:lookback_terms]:
        rates.append(sec_terms[t] if t in sec_terms else course_terms[t])
    return float(np.mean(rates)) if rates else np.nan
# Cache for professor ratings to avoid repeated lookups
_PROF_DF_CACHE = None

# 3. Professor/Class rating lookup
def _load_professors(data_dir=DATA_DIR):
    global _PROF_DF_CACHE
    if _PROF_DF_CACHE is None:
        path = os.path.join(data_dir, "sfu_rmp", "sfu_professors.csv")
        if os.path.exists(path):
            _PROF_DF_CACHE = pd.read_csv(path)
        else:
            _PROF_DF_CACHE = pd.DataFrame()
    return _PROF_DF_CACHE

# Returns the average professor rating for a given course (e.g. "CMPT225"), averaged across every professor who's ever taught it.
def get_course_rating(course, data_dir=DATA_DIR):
    df = _load_professors(data_dir)
    if df.empty:
        return np.nan
    target = normalize_token(course)
    mask = df["courses"].fillna("").apply(
        lambda s: target in [normalize_token(c) for c in s.split(";")]
    )
    matches = df.loc[mask, "rating"]
    if len(matches):
        return float(matches.mean())
    return float(df["rating"].mean())
# Instructor-specific rating via the SFU course outlines API.
BASE_URL = "http://www.sfu.ca/bin/wcm/course-outlines"

# Offerings does not include instructors, query SFU API by dept/course/section/year/term
def get_section_details(dept, course_num, section, year, term):
    url = f"{BASE_URL}?{year}/{term}/{dept}/{course_num}/{section}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            # No outline on file for this section/term. This is a normal, expected outcome, not an actual error, so don't log as one
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Failed: {dept} {course_num} {section} - {e}")
        return None
    
# SFU course-outline responses put instructor info either directly under 'instructor' or nested under 'info' -> 'instructor'
def _extract_instructor_names(details):
    if not isinstance(details, dict):
        return []
    instr = details.get("instructor")
    if not isinstance(instr, list):
        info = details.get("info")
        instr = info.get("instructor") if isinstance(info, dict) else None
    if not isinstance(instr, list):
        return []
    return [e["name"] for e in instr if isinstance(e, dict) and e.get("name")]

# Simplify instructor names, ex. comparing "John Smith" vs "Smith, J."
def _normalize_name_tokens(name):
    return frozenset(t for t in re.split(r"[^A-Za-z]+", str(name).upper()) if t)

# Comparison function for names
def _names_match(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return False
    return a_tokens == b_tokens or len(a_tokens & b_tokens) >= 2
#Cache ratings to avoid repeated lookups for the same section
_INSTRUCTOR_RATING_CACHE = {}

# Returns the RMP rating of whoever actually taught this section (via the SFU course outlines API), or None if it can't be found.
def get_instructor_rating(dept, course_num, section, year, term, data_dir=DATA_DIR):
    key = (str(dept).upper(), str(course_num).upper(), str(section).upper(), year, term)
    if key in _INSTRUCTOR_RATING_CACHE:
        return _INSTRUCTOR_RATING_CACHE[key]
    details = get_section_details(dept, course_num, section, year, term)
    names = _extract_instructor_names(details)
    prof_df = _load_professors(data_dir)
    rating = None
    if names and not prof_df.empty:
        prof_tokens = [(_normalize_name_tokens(n), r) for n, r in zip(prof_df["name"], prof_df["rating"])]
        matched = [r for nm in names for tok, r in prof_tokens if _names_match(_normalize_name_tokens(nm), tok)]
        if matched:
            rating = float(np.mean(matched))
    _INSTRUCTOR_RATING_CACHE[key] = rating
    return rating

# Program-requirement lookup from course planners, if major requirement or prerequiste
def get_course_popularity_metrics(course, data_dir=DATA_DIR):
    folder = os.path.join(data_dir, "coursePlanners")
    target = normalize_token(course)
    count = 0
    if os.path.isdir(folder):
        for planner in os.listdir(folder):
            path = os.path.join(folder, planner)
            # Only read through .txt files
            if not path.lower().endswith(".txt"):
                continue
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if target in [normalize_token(c) for c in content.splitlines()]:
                        count += 1
            except Exception as e:
                print(f"Warning: Failed to read planner file {path}: {e}")
                continue
    # Returns (is_major_requirement, num_planners_containing_course).
    return (1 if count > 0 else 0), count
# Based off data and analysis to predict fullness
FEATURE_NAMES = ["week_of_reg", "max_enrol", "sem_taught_2yrs", "course_rating",
                  "prof_rating", "is_major_req", "historic_fill_rate"]

# Build the training set + decision tree
def build_training_set(offerings_df, fill_df, data_dir=DATA_DIR, lookback_terms=4,
                        use_instructor_api=True, max_instructor_lookups=300):
    # Build the feature matrix X and target vector y from the historical offerings DataFrame.
    df = offerings_df.copy()
    df["code"] = df.apply(lambda r: normalize_code(r["subject"], r["catnbr"]), axis=1)
    # cache expensive per-course lookups
    codes = df["code"].unique()
    course_rating_map = {c: get_course_rating(c, data_dir) for c in codes}
    major_map = {c: get_course_popularity_metrics(c, data_dir)[0] for c in codes}
    df["course_rating"] = df["code"].map(course_rating_map)
    df["is_major_req"] = df["code"].map(major_map)
    course_idx, section_idx = _build_fill_week_index(fill_df)
    df["historic_fill_rate"] = df.apply(
        lambda r: get_historic_fill_rate(course_idx, section_idx, r["subject"], r["catnbr"],
                                          r["section"], r["week_of_reg"], r["term_code"], lookback_terms),
        axis=1,
    )
    # prof_rating: instructor-specific rating via the SFU course outlines API
    df["prof_rating"] = np.nan
    if use_instructor_api:
        combos = df[["subject", "catnbr", "section", "term_code"]].drop_duplicates()
        if len(combos) > max_instructor_lookups:
            print(f" {len(combos)} unique sections found; only querying the SFU course-outlines "
                  f"API for the first {max_instructor_lookups} (raise with --max-instructor-lookups). "
                  f"The rest fall back to course_rating.")
        prof_map = {}
        n_failed = 0
        for _, r in combos.head(max_instructor_lookups).iterrows():
            term_name, term_year = term_name_from_code(r["term_code"])
            rating = get_instructor_rating(r["subject"], r["catnbr"], r["section"], term_year, term_name, data_dir)
            prof_map[(r["subject"], r["catnbr"], r["section"], r["term_code"])] = rating
            if rating is None:
                n_failed += 1
        print(f" Instructor lookups: {len(prof_map) - n_failed}/{len(prof_map)} resolved to a rating.")
        df["prof_rating"] = df.apply(
            lambda r: prof_map.get((r["subject"], r["catnbr"], r["section"], r["term_code"])), axis=1
        )
    df = df.dropna(subset=["week_of_reg", "max_enrol", "percent_filled"])
    df["sem_taught_2yrs"] = df["sem_taught_2yrs"].fillna(0)
    df["course_rating"] = df["course_rating"].fillna(df["course_rating"].mean())
    # No instructor match (API skipped/failed/no RMP hit), fall back to course_rating
    df["prof_rating"] = df["prof_rating"].fillna(df["course_rating"])
    fill_default = df["historic_fill_rate"].mean()
    if pd.isna(fill_default):
        fill_default = df["percent_filled"].mean()
    df["historic_fill_rate"] = df["historic_fill_rate"].fillna(fill_default)
    X = df[FEATURE_NAMES].astype(float).values
    y = df["percent_filled"].astype(float).values
    return X, y, df

def build_and_train_decision_tree(X, y):
    # Use model DecisionTreeRegressor with hyperparameters tuned for this dataset. The model is trained on the provided feature matrix X and target vector y.
    model = DecisionTreeRegressor(max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=42)
    model.fit(X, y)
    return model

def evaluate_split(X, y, test_size):
    # Splits dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    #Train model
    model = build_and_train_decision_tree(X_train, y_train)
    preds = model.predict(X_test)
    #Evaluate performance
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2, model

def compare_test_splits(X, y, test_sizes):
    # Compares different test sizes to default to best
    results = []
    for ts in test_sizes:
        mse, r2, model = evaluate_split(X, y, ts)
        results.append((ts, mse, r2, model))
    results.sort(key=lambda t: -t[2])  # best R2 first
    best_ts, best_mse, best_r2, best_model = results[0]
    print(f"Split Comparison: Best test_size = {best_ts} (R2={best_r2:.4f}, RMSE={best_mse ** 0.5:.4f})")
    for i, (ts, mse, r2, _) in enumerate(results):
        n_test = int(round(len(X) * ts))
        n_train = len(X) - n_test
        marker = " <- best" if i == 0 else ""
        print(f"{'':>19}{ts:>10.2f} {n_train:>10} {n_test:>10} {mse:>10.4f} {mse ** 0.5:>10.4f} {r2:>10.4f}{marker}")
    for name, imp in sorted(zip(FEATURE_NAMES, best_model.feature_importances_), key=lambda t: -t[1]):
        print(f"    feature importance: {name:<16} {imp:.3f}")
    return best_ts, best_mse, best_r2

# Prediction for a specific course/section/enrollment date
def get_actual_snapshot(offerings_df, subject, catnbr, section, term_code, enrollment_date):
    # Returns the most recent historical snapshot of a course section on or before the enrollment date.
    sub = offerings_df[
        (offerings_df["term_code"] == term_code)
        & (offerings_df["subject"].astype(str).str.upper() == subject.upper())
        & (offerings_df["catnbr"].astype(str).str.upper() == str(catnbr).upper())
        & (offerings_df["section"].astype(str).str.upper() == section.upper())
        & (offerings_df["date"] <= enrollment_date)
    ]
    if sub.empty:
        return None
    row = sub.sort_values("date").iloc[-1]
    return row

# Predict the fullness of a course section based on historical data and features
def predict_fullness(model, offerings_df, fill_df, course, section, enrollment_date_str,
                      data_dir=DATA_DIR, lookback_terms=4, use_instructor_api=True):
    term_name, term_year, week_of_reg, term_code, reg_open = parse_enrollment_date(enrollment_date_str)
    print(f"1. Parsed date -> Term: {term_name.title()} {term_year} (code {term_code}), "
          f"Week of registration: {week_of_reg} (opens {reg_open.date()})")
    m = re.match(r"([A-Za-z]+)\s*([0-9A-Za-z]+)", course)
    subject, catnbr = (m.group(1), m.group(2)) if m else (course, "")
    code = normalize_code(subject, catnbr)
    enrollment_date = datetime.strptime(enrollment_date_str, "%Y-%m-%d")
    snapshot = get_actual_snapshot(offerings_df, subject, catnbr, section, term_code, enrollment_date)
    # If a snapshot exists, use its max_enrol and sem_taught_2yrs, otherwise, fall back to department median values.
    if snapshot is not None:
        max_enrol = snapshot["max_enrol"]
        sem_taught_2yrs = snapshot["sem_taught_2yrs"] or 0
        print(f"2. Found real offering data for {code} {section} (term {term_code}): "
              f"MaxEnrol={max_enrol}, last known %Filled on/before date={snapshot['percent_filled']:.2%} "
              f"(as of {snapshot['date'].date()})")
    else:
        hist = offerings_df[offerings_df["subject"].astype(str).str.upper() == subject.upper()]
        max_enrol = hist["max_enrol"].median() if not hist.empty else offerings_df["max_enrol"].median()
        sem_taught_2yrs = hist["sem_taught_2yrs"].median() if not hist.empty else 0
        print(f"2. No exact historical record for {code} {section} in term {term_code} — "
              f"using department-median MaxEnrol={max_enrol:.0f} as a stand-in.")

    course_rating = get_course_rating(code, data_dir)
    print(f"3a. Course-level rating for {code} (average across every professor who's taught it): "
          f"{course_rating:.2f}/5.0")
    prof_rating = get_instructor_rating(subject, catnbr, section, term_year, term_name, data_dir) \
        if use_instructor_api else None
    # If the instructor-specific rating is not found, fall back to the course-level rating.
    if prof_rating is not None:
        print(f"3b. Instructor-specific rating for {code} {section} "
              f"(via SFU course outlines API): {prof_rating:.2f}/5.0")
    else:
        prof_rating = course_rating
        reason = "lookup skipped (--skip-instructor-api)" if not use_instructor_api else \
                 "no outline/instructor/RMP match found"
        print(f"3b. No instructor-specific rating for {code} {section} ({reason}) — "
              f"falling back to the course-level rating.")

    course_idx, section_idx = _build_fill_week_index(fill_df)
    historic_fill = get_historic_fill_rate(course_idx, section_idx, subject, catnbr, section,
                                            week_of_reg, term_code, lookback_terms)
    # If no historic week-by-week data is found, fall back to the overall average fill rate across all courses in the dataset
    if not np.isnan(historic_fill):
        print(f"3c. Historic fill rate for {code} at week {week_of_reg} of registration "
              f"(averaged over up to {lookback_terms} prior terms in data/courseFillByWeek/): "
              f"{historic_fill:.1%}")
    else:
        historic_fill = fill_df["percent_filled"].mean() if not fill_df.empty else 0.5
        print(f"3c. No historic week-by-week data found for {code} at week {week_of_reg} — "
              f"using the overall average ({historic_fill:.1%}) as a stand-in.")
    is_major, n_planners = get_course_popularity_metrics(code, data_dir)
    print(f"4. Curriculum metrics -> Program requirement: {bool(is_major)} "
          f"(appears in {n_planners} planner file(s))")
    feature_vector = np.array([[week_of_reg, max_enrol, sem_taught_2yrs, course_rating,
                                 prof_rating, is_major, historic_fill]])
    predicted = model.predict(feature_vector)[0]
    predicted = min(max(predicted, 0.0), 1.0)
    print(f"5. PREDICTED CLASS FULLNESS: {predicted * 100:.1f}%")
    return predicted

def main():
    # User arguments for training the decision tree model
    parser = argparse.ArgumentParser(description="Train the course-fullness decision tree and save it for reuse.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH,
                         help=f"Where to save the trained bundle. Default {MODEL_PATH}")
    parser.add_argument("--test-sizes", default="0.2",
                         help="Comma-separated train/test split ratios to compare, e.g. '0.1,0.2,0.3'. "
                              "The best-scoring one is reported; the saved model is always refit on "
                              "all data regardless. Default '0.2'.")
    parser.add_argument("--lookback-terms", type=int, default=4,
                         help="How many previous terms in data/courseFillByWeek/ to average the "
                              "same-week fill rate over. Default 4.")
    parser.add_argument("--skip-instructor-api", action="store_true",
                         help="Skip the SFU course-outlines API lookup (faster, works offline); "
                              "falls back to the course-level rating everywhere.")
    parser.add_argument("--max-instructor-lookups", type=int, default=300,
                         help="Cap on unique sections queried against the SFU course-outlines API "
                              "while building the training set, to limit runtime. Default 300.")
    args = parser.parse_args()
    try:
        test_sizes = [float(t) for t in args.test_sizes.split(",") if t.strip()]
    except ValueError:
        print(f"Error: --test-sizes must be a comma-separated list of numbers, got '{args.test_sizes}'")
        sys.exit(1)
    print(f"Using data directory: {args.data_dir}")
    offerings_df = load_all_offerings(args.data_dir)
    if offerings_df.empty:
        print(f"Error: No course offering files found under {args.data_dir}/courseOfferings/. "
              f"Add at least one <termCode>.xlsx report there and try again.")
        sys.exit(1)
    print(f"Loaded {len(offerings_df)} historical (date, section) snapshots "
          f"across terms: {sorted(offerings_df['term_code'].unique().tolist())}")
    fill_df = load_course_fill_by_week(args.data_dir)
    if fill_df.empty:
        print(f"Warning: No files found under {args.data_dir}/courseFillByWeek/ — "
              f"historic_fill_rate will fall back to an overall average.")
    else:
        print(f"Loaded {len(fill_df)} courseFillByWeek snapshots "
              f"across terms: {sorted(fill_df['term_code'].unique().tolist())}")
    X, y, _ = build_training_set(offerings_df, fill_df, args.data_dir, lookback_terms=args.lookback_terms,
                                  use_instructor_api=not args.skip_instructor_api,
                                  max_instructor_lookups=args.max_instructor_lookups)
    print(f"Built {len(X)} training rows with features {FEATURE_NAMES}")
    # Save best test size
    best_ts, best_mse, best_r2 = compare_test_splits(X, y, test_sizes)
    print(f"Training final model on split {best_ts:.0%}/{1 - best_ts:.0%} and saving to {args.model_path}...")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=best_ts, random_state=42)
    model = build_and_train_decision_tree(X_train, y_train)
    bundle = {
        "model": model,
        "offerings_df": offerings_df,
        "fill_df": fill_df,
        "feature_names": FEATURE_NAMES,
        "lookback_terms": args.lookback_terms,
        "use_instructor_api": not args.skip_instructor_api,
        "data_dir": args.data_dir,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.model_path)) or ".", exist_ok=True)
    joblib.dump(bundle, args.model_path, compress=3)
    print(f"Saved trained model bundle to {args.model_path}")
    print(f"Run predictions with: python decisionTree.py <course> <section> <enrollment_date>")
if __name__ == "__main__":
    main()