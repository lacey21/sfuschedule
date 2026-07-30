import requests
import pandas as pd
import time
import re
import os
import sys

BASE_URL = "http://www.sfu.ca/bin/wcm/course-outlines"

# every term we have a raw enrollment database export for, and the (year, term)
# the course-outlines API expects for each. SFU's export column names changed
# starting with term 1247 (Sect/Credit -> Section/Units), handled below.
TERM_CODES = {
    '1214': (2021, 'summer'),
    '1217': (2021, 'fall'),
    '1221': (2022, 'spring'),
    '1224': (2022, 'summer'),
    '1227': (2022, 'fall'),
    '1231': (2023, 'spring'),
    '1234': (2023, 'summer'),
    '1237': (2023, 'fall'),
    '1241': (2024, 'spring'),
    '1244': (2024, 'summer'),
    '1247': (2024, 'fall'),
    '1251': (2025, 'spring'),
    '1254': (2025, 'summer'),
    '1261': (2026, 'spring'),
}


def enrollment_file_for(term_code, data_dir):
    """Raw exports are named database.<code>.xlsx for older terms and
    database-<code>.xlsx for newer ones; check both."""
    for name in (f"database-{term_code}.xlsx", f"database.{term_code}.xlsx"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No raw enrollment export found for term {term_code} in {data_dir}")


def load_offered_courses(enrollment_path):
    """
    Loads unique (subject, catalog #, section, units) combos from a raw SFU
    enrollment database export (.xlsx report, with 6 title rows before the
    header, or a plain .csv). Normalizes the two column-naming conventions
    SFU has used across terms (Sect/Credit vs Section/Units).
    """
    if enrollment_path.endswith('.xlsx'):
        df = pd.read_excel(enrollment_path, header=6)
    else:
        df = pd.read_csv(enrollment_path)

    section_col = 'Section' if 'Section' in df.columns else 'Sect'
    units_col = 'Units' if 'Units' in df.columns else 'Credit'

    combos = df[['Subject', 'CatNbr', section_col, units_col]].drop_duplicates(
        subset=['Subject', 'CatNbr', section_col]
    )
    combos = combos.rename(columns={section_col: 'Section', units_col: 'Units'})
    return combos.to_dict('records')


def extract_prereq_codes(prereq_text):
    if not prereq_text or not isinstance(prereq_text, str):
        return [], False
    required_text = re.split(r'recommended\s*:', prereq_text, flags=re.IGNORECASE)[0]
    codes = re.findall(r'\b([A-Z]{2,4})\s?(\d{3}[A-Z]?)\b', required_text)
    codes = [f"{dept} {num}" for dept, num in codes]
    unparseable = len(codes) == 0 and bool(re.search(
        r'permission|waiver|co-?op|instructor', prereq_text, re.IGNORECASE))
    return list(dict.fromkeys(codes)), unparseable


def get_section_details(dept, course_num, section, year, term):
    url = f"{BASE_URL}?{year}/{term}/{dept}/{course_num}/{section}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  Failed: {dept} {course_num} {section} - {e}")
        return None


def discover_offered_sections(year, term):
    """
    Crawl the course-outlines API's own hierarchy (dept -> course -> section)
    to find every section offered in a term. Used for terms with no
    historical enrollment-database export to read the offered-sections list
    from - e.g. a future/current term SFU hasn't run an enrollment report for
    yet, but has already published outlines for.
    """
    combos = []
    depts = requests.get(f"{BASE_URL}?{year}/{term}", timeout=10).json()
    print(f"Found {len(depts)} departments for {term} {year}")
    for d in depts:
        dept = d['value']
        try:
            courses = requests.get(f"{BASE_URL}?{year}/{term}/{dept}", timeout=10).json()
        except Exception as e:
            print(f"  Failed to list courses for {dept} - {e}")
            continue
        for c in courses:
            course_num = c['value']
            try:
                sections = requests.get(
                    f"{BASE_URL}?{year}/{term}/{dept}/{course_num}", timeout=10
                ).json()
            except Exception as e:
                print(f"  Failed to list sections for {dept} {course_num} - {e}")
                continue
            if not isinstance(sections, list):
                # the API returns an error object (not a list) for a handful of
                # courses without published sections yet, e.g. directed-studies
                # or special-topics numbers - nothing to schedule, so skip
                continue
            for s in sections:
                combos.append({
                    'Subject': dept.upper(),
                    'CatNbr': course_num.upper(),
                    'Section': s['value'].upper(),
                })
            time.sleep(0.05)
        time.sleep(0.05)
    print(f"Found {len(combos)} unique sections for {term} {year}")
    return combos


def parse_instructors(instructor_list):
    """Extract deduped instructor display names from the API's 'instructor' list."""
    names = []
    for entry in instructor_list or []:
        name = entry.get('name') or f"{entry.get('firstName', '')} {entry.get('lastName', '')}".strip()
        if name:
            names.append(name)
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def parse_schedule(course_schedule):
    seen = set()
    slots = []
    for entry in course_schedule:
        key = (entry.get('days'), entry.get('startTime'), entry.get('endTime'))
        if key in seen or not all(key):
            continue
        seen.add(key)
        slots.append({
            'days': entry.get('days'),
            'startTime': entry.get('startTime'),
            'endTime': entry.get('endTime'),
            'campus': entry.get('campus'),
        })
    return slots


def get_prerequisites_and_schedule(combos, year, term, output_csv):
    rows = []
    for i, combo in enumerate(combos):
        dept = str(combo['Subject']).strip()
        course_num = str(combo['CatNbr']).strip()
        section = str(combo['Section']).strip()

        if i == 0 or (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(combos)}] {dept} {course_num} {section}")
        details = get_section_details(dept, course_num, section, year, term)
        if not details:
            continue

        info = details.get('info', {})
        # SFU wraps long prerequisite text with literal CRLFs; collapse to
        # single-line so the CSV doesn't end up with mixed line endings.
        prereq_text = ' '.join(info.get('prerequisites', '').split())
        prereq_codes, unparseable = extract_prereq_codes(prereq_text)
        schedule = parse_schedule(details.get('courseSchedule', []))
        instructors = parse_instructors(details.get('instructor', []))

        rows.append({
            'course': f"{dept} {course_num}",
            'section': section,
            'term': term,
            'year': year,
            'prereq_text': prereq_text,
            'prereq_codes': ';'.join(prereq_codes),
            'prereq_unparseable': unparseable,
            'schedule': schedule,
            'instructors': ';'.join(instructors),
            'units': combo.get('Units') or info.get('units', ''),
        })
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    return df


def scrape_term_from_enrollment_file(enrollment_path, year, term, output_csv):
    combos = load_offered_courses(enrollment_path)
    print(f"Found {len(combos)} unique sections in {enrollment_path}")
    return get_prerequisites_and_schedule(combos, year, term, output_csv)


def scrape_term_from_api(year, term, output_csv):
    combos = discover_offered_sections(year, term)
    return get_prerequisites_and_schedule(combos, year, term, output_csv)


def scrape_historical_terms(data_dir):
    """Runs the deferred batch over TERM_CODES (terms 1214-1261), skipping
    any that already have an output CSV."""
    for term_code, (year, term) in TERM_CODES.items():
        output_csv = f"{data_dir}/sfu_prereqs_and_schedule_{term_code}.csv"
        if os.path.exists(output_csv):
            print(f"Skipping {term_code} ({term} {year}) - already scraped")
            continue
        enrollment_path = enrollment_file_for(term_code, data_dir)
        print(f"=== Scraping {term_code} ({term} {year}) from {enrollment_path} ===")
        scrape_term_from_enrollment_file(enrollment_path, year, term, output_csv)


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(_script_dir, "..", "data", "courseOfferings")

    if len(sys.argv) > 1 and sys.argv[1] == "historical":
        scrape_historical_terms(data_dir)
    else:
        output_csv = f"{data_dir}/sfu_prereqs_and_schedule_1267.csv"
        print("=== Scraping 1267 (fall 2026) from the live course-outlines API ===")
        scrape_term_from_api(2026, "fall", output_csv)
