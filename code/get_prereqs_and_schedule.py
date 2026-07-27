import requests
import pandas as pd
import time
import re

BASE_URL = "http://www.sfu.ca/bin/wcm/course-outlines"

def load_offered_courses(enrollment_csv):
    df = pd.read_csv(enrollment_csv)
    combos = df[['Subject', 'CatNbr', 'Section']].drop_duplicates()
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


def get_prerequisites_and_schedule(enrollment_csv, year, term, output_csv):
    combos = load_offered_courses(enrollment_csv)
    print(f"Found {len(combos)} unique sections in {enrollment_csv}")

    rows = []
    for i, combo in enumerate(combos):
        dept = str(combo['Subject']).strip()
        course_num = str(combo['CatNbr']).strip()
        section = str(combo['Section']).strip()

        print(f"[{i+1}/{len(combos)}] {dept} {course_num} {section}")
        details = get_section_details(dept, course_num, section, year, term)
        if not details:
            continue

        info = details.get('info', {})
        prereq_text = info.get('prerequisites', '')
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
        })
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    return df


if __name__ == "__main__":
    get_prerequisites_and_schedule(
        enrollment_csv="data/courseOfferings/database.1247.csv",
        year="2024",
        term="fall",
        output_csv="data/courseOfferings/sfu_prereqs_and_schedule_1247.csv",
    )
