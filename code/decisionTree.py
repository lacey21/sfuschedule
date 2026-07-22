#!/usr/bin/env python3
"""
decisionTree.py — predicts how full a course section will be at a given
enrollment (registration) date, using SFU Institutional Research & Planning
"Course Section Availability" reports as historical training data.

Usage:
    python decisionTree.py <course> <section> <enrollment_date YYYY-MM-DD>

Example:
    python decisionTree.py CMPT225 D100 2026-07-15
"""

import sys
import os
import re
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import openpyxl
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Data directory is sfuschedule/data, current file is in sfuschedule/code, so we need to go up one level to get to data
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

REG_OPEN = {
    "spring": (11, 13),  # opens in the prior calendar year
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

    week_of_registration = max(1, (date_obj - reg_open).days // 7 + 1)
    term_code = (term_year - 1900) * 10 + TERM_DIGIT[term_name]

    return term_name, term_year, week_of_registration, term_code, reg_open


def _week_of_registration_for_date(date_obj, term_name, term_year):
    if term_name == "spring":
        reg_open = datetime(term_year - 1, *REG_OPEN["spring"])
    elif term_name == "summer":
        reg_open = datetime(term_year, *REG_OPEN["summer"])
    else:
        reg_open = datetime(term_year, *REG_OPEN["fall"])
    return max(1, (date_obj - reg_open).days // 7 + 1)


def term_name_from_code(term_code):
    digit = term_code % 10
    year = 1900 + term_code // 10
    name = {1: "spring", 4: "summer", 7: "fall"}.get(digit, "unknown")
    return name, year


def normalize_code(subject, catnbr):
    return f"{str(subject).strip().upper()}{str(catnbr).strip().upper()}"


def normalize_token(token):
    return re.sub(r"[\s\-]", "", str(token)).strip().upper()


# 2. Load a term's course-offering file (either report format) into a
#    normalized long-format DataFrame.
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
    # Parses the 'pivot.percent.filled' sheet layout: one row per section,
    # %Filled spread across date columns, with forward-filled course info.
    header_row = None
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=20, values_only=True), start=1):
        if row and row[0] and str(row[0]).strip().casefold() == "subject":
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()

    week_label_row = [r for r in ws.iter_rows(min_row=header_row - 2, max_row=header_row - 2, values_only=True)][0]
    header = [r for r in ws.iter_rows(min_row=header_row, max_row=header_row, values_only=True)][0]

    # Case-insensitive lookup so a slightly renamed header (e.g. 'Section'
    # instead of 'Sect') doesn't crash the loader — just falls back to None.
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


def _parse_percent(x):
    """Handles both raw fractions (0.53) and percent-strings ('53%')."""
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


def _parse_day_month(s):
    # Parses a string like "04-Jul" or "4 Jul" into (day, month) pair
    m = re.match(r"\s*(\d{1,2})[\s\-]?([A-Za-z]{3})", str(s))
    if not m:
        return None
    month = _MONTH_ABBR.get(m.group(2).lower())
    if month is None:
        return None
    return int(m.group(1)), month


def _load_csv_long_format(path, term_code):
    # Loads a CSV file in the long-format layout (one row per date/section) into a DataFrame.
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


def load_offering_file(path, term_code):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _load_csv_long_format(path, term_code)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if "database" in wb.sheetnames:
        return _load_long_format(wb["database"], term_code)
    return _load_pivot_format(wb[wb.sheetnames[0]], term_code)


def _extract_term_code(fname):
    # Extracts a 4-digit term code from the filename (e.g. "20231" for Spring 2023).
    matches = re.findall(r"(?<!\d)(1\d{3})(?!\d)", fname)
    return int(matches[0]) if matches else None


def load_all_offerings(data_dir=DATA_DIR):
    # Loads all course-offering files in the data/courseOfferings/ folder into a single DataFrame.
    folder = os.path.join(data_dir, "courseOfferings")
    frames = []
    if not os.path.isdir(folder):
        return pd.DataFrame()
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".xlsx", ".xls", ".csv")):
            continue
        term_code = _extract_term_code(fname)
        if term_code is None:
            print(f"[WARN] Skipping {fname}: couldn't find a 4-digit term code in the filename.")
            continue
        df = load_offering_file(os.path.join(folder, fname), term_code)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    all_df["percent_filled"] = pd.to_numeric(all_df["percent_filled"], errors="coerce")
    all_df["max_enrol"] = pd.to_numeric(all_df["max_enrol"], errors="coerce")
    all_df["sem_taught_2yrs"] = pd.to_numeric(all_df["sem_taught_2yrs"], errors="coerce")
    all_df = all_df.dropna(subset=["percent_filled"])
    return all_df

# 3. Professor/Class rating lookup 
_PROF_DF_CACHE = None

def _load_professors(data_dir=DATA_DIR):
    global _PROF_DF_CACHE
    if _PROF_DF_CACHE is None:
        path = os.path.join(data_dir, "sfu_rmp", "sfu_professors.csv")
        if os.path.exists(path):
            _PROF_DF_CACHE = pd.read_csv(path)
        else:
            _PROF_DF_CACHE = pd.DataFrame()
    return _PROF_DF_CACHE


def get_professor_rating(course, data_dir=DATA_DIR):
    # Returns the average professor rating for a given course (e.g. "CMPT225") based on the RMP data.
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


# 4. Program-requirement lookup from course planners
def get_course_popularity_metrics(course, data_dir=DATA_DIR):
    """Returns (is_major_requirement, num_planners_containing_course)."""
    folder = os.path.join(data_dir, "coursePlanners")
    target = normalize_token(course)
    count = 0
    if os.path.isdir(folder):
        for planner in os.listdir(folder):
            path = os.path.join(folder, planner)
            print(f"[DEBUG] Checking planner file: {path}")
            #Example, ENSC COMPUTER planner (Spring 2023 onward) PDF.pdf
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if target in [normalize_token(c) for c in content.splitlines()]:
                        count += 1
            except Exception as e:
                print(f"[WARN] Failed to read planner file {path}: {e}")
                continue
    return (1 if count > 0 else 0), count


# 5. Build the training set + decision tree
FEATURE_NAMES = ["week_of_reg", "max_enrol", "sem_taught_2yrs", "prof_rating", "is_major_req"]


def build_training_set(offerings_df, data_dir=DATA_DIR):
    # Build the feature matrix X and target vector y from the historical offerings DataFrame.
    df = offerings_df.copy()
    df["code"] = df.apply(lambda r: normalize_code(r["subject"], r["catnbr"]), axis=1)

    # cache expensive per-course lookups
    codes = df["code"].unique()
    prof_rating_map = {c: get_professor_rating(c, data_dir) for c in codes}
    major_map = {c: get_course_popularity_metrics(c, data_dir)[0] for c in codes}

    df["prof_rating"] = df["code"].map(prof_rating_map)
    df["is_major_req"] = df["code"].map(major_map)

    df = df.dropna(subset=["week_of_reg", "max_enrol", "percent_filled"])
    df["sem_taught_2yrs"] = df["sem_taught_2yrs"].fillna(0)
    df["prof_rating"] = df["prof_rating"].fillna(df["prof_rating"].mean())

    X = df[FEATURE_NAMES].astype(float).values
    y = df["percent_filled"].astype(float).values
    return X, y, df


def build_and_train_decision_tree(X, y):
    model = DecisionTreeRegressor(max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=42)
    model.fit(X, y)
    return model


def validate_model(model, X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[VALIDATION] n_train={len(X_train)} n_test={len(X_test)} "
          f"MSE={mse:.4f} RMSE={mse ** 0.5:.4f} R2={r2:.4f}")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1]):
        print(f"    feature importance: {name:<16} {imp:.3f}")
    # refit on all data for the final model used to predict
    model.fit(X, y)
    return model


# 6. Prediction for a specific course/section/enrollment date
def get_actual_snapshot(offerings_df, subject, catnbr, section, term_code, enrollment_date):
    """If we happen to already have real historical data for this exact
    term/course/section, return the closest known %Filled on/before the
    requested date (useful for sanity-checking predictions)."""
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
def predict_fullness(model, offerings_df, course, section, enrollment_date_str, data_dir=DATA_DIR):
    term_name, term_year, week_of_reg, term_code, reg_open = parse_enrollment_date(enrollment_date_str)
    print(f"1. Parsed date -> Term: {term_name.title()} {term_year} (code {term_code}), "
          f"Week of registration: {week_of_reg} (opens {reg_open.date()})")

    m = re.match(r"([A-Za-z]+)\s*([0-9A-Za-z]+)", course)
    subject, catnbr = (m.group(1), m.group(2)) if m else (course, "")
    code = normalize_code(subject, catnbr)

    enrollment_date = datetime.strptime(enrollment_date_str, "%Y-%m-%d")
    snapshot = get_actual_snapshot(offerings_df, subject, catnbr, section, term_code, enrollment_date)

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

    prof_rating = get_professor_rating(code, data_dir)
    print(f"3. Professor rating for {code} (course-level average, no instructor field in the "
          f"source reports): {prof_rating:.2f}/5.0")

    is_major, n_planners = get_course_popularity_metrics(code, data_dir)
    print(f"4. Curriculum metrics -> Program requirement: {bool(is_major)} "
          f"(appears in {n_planners} planner file(s))")

    feature_vector = np.array([[week_of_reg, max_enrol, sem_taught_2yrs, prof_rating, is_major]])
    predicted = model.predict(feature_vector)[0]
    predicted = min(max(predicted, 0.0), 1.0)
    print(f"5. PREDICTED CLASS FULLNESS: {predicted * 100:.1f}%")
    return predicted


def main():
    parser = argparse.ArgumentParser(description="Predict course section fullness at a given enrollment date.")
    parser.add_argument("course", help="e.g. CMPT225 or 'CMPT 225'")
    parser.add_argument("section", help="e.g. D100")
    parser.add_argument("enrollment_date", help="YYYY-MM-DD")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--test-size", type=float, default=0.2,
                         help="Fraction of the historical data held out for validation (0-1). Default 0.2.")
    args = parser.parse_args()

    print(f"Running Prediction for {args.course} {args.section} on {args.enrollment_date} ---")
    print(f"Using data directory: {args.data_dir}")
    offerings_df = load_all_offerings(args.data_dir)
    if offerings_df.empty:
        print(f"ERROR: No course offering files found under {args.data_dir}/courseOfferings/. "
              f"Add at least one <termCode>.xlsx report there and try again.")
        sys.exit(1)
    print(f"Loaded {len(offerings_df)} historical (date, section) snapshots "
          f"across terms: {sorted(offerings_df['term_code'].unique().tolist())}")

    X, y, _ = build_training_set(offerings_df, args.data_dir)
    print(f"Built {len(X)} training rows with features {FEATURE_NAMES}")

    model = build_and_train_decision_tree(X, y)
    model = validate_model(model, X, y, test_size=args.test_size)

    predict_fullness(model, offerings_df, args.course, args.section, args.enrollment_date, args.data_dir)


if __name__ == "__main__":
    main()