"""
scheduler.py

Core scheduling logic for the SFU Degree Schedule Helper Minimal Viable System (MVS).
Check the MVS flow diagram I(Dora) have in the Google Docs to see the flow!

"""

import ast
import re
import sys
import os
from datetime import datetime
import pandas as pd

from parse_transcript import parse_sfu_transcript
import professor_ratings

# resolve data/ relative to this file's location rather than assuming the caller's cwd so that `python3 scheduler.py ...` works whether run from sfuschedule/ or code/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data", "courseOfferings")


def parse_enrollment_date(date_str):
    """
    Parse a user-entered enrollment date string into a datetime.date.
    Accepts a few common formats so we're not overly strict about how
    the student types it in.
    """
    date_str = date_str.strip()
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Couldn't parse date '{date_str}'. Try formats like 2026-09-15 or 09/15/2026."
    )


def get_term_code(year, term):
    """
    Convert year + term name into SFU's term code format (1YYT), e.g.
    (2025, 'fall') -> '1257'. Follows SFU's convention: 1=spring, 4=summer, 7=fall.
    """
    term_map = {'spring': '1', 'summer': '4', 'fall': '7'}
    term_key = term.strip().lower()
    if term_key not in term_map:
        raise ValueError(f"Unknown term '{term}'. Expected spring, summer, or fall.")
    yy = str(year)[-2:]
    return f"1{yy}{term_map[term_key]}"


def enrollment_date_to_term(enrollment_date):
    """
    Given a datetime/date for when the student will enroll, pick the
    target term:
      Jan-Apr -> that year's Spring
      May-Aug -> that year's Summer
      Sep-Dec -> that year's Fall
      (i think this is supposed to be staggered but I'll leave it like this for now and handle the logic later on)
    """
    month = enrollment_date.month
    year = enrollment_date.year
    if month <= 4:
        term = 'spring'
    elif month <= 8:
        term = 'summer'
    else:
        term = 'fall'
    return year, term


def load_data(transcript_result, data_dir=DEFAULT_DATA_DIR, year=None, term=None,
              enrollment_date=None):
    """
    Loads completed courses + major from a parsed transcript result, and the
    matching term's prereq+schedule CSV. Provide either (year, term) directly,
    or an enrollment_date to derive them.

    *** Returns a dict: {major, cgpa, courses_taken, course_data, term_code}.
    """
    if enrollment_date is not None:
        year, term = enrollment_date_to_term(enrollment_date)
    if year is None or term is None:
        raise ValueError("Must provide either (year, term) or enrollment_date")

    term_code = get_term_code(year, term)
    csv_path = f"{data_dir}/sfu_prereqs_and_schedule_{term_code}.csv"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"\nNo scraped course data found for {term.capitalize()} {year} "
            f"(term code {term_code}).\n"
            f"Expected file: {csv_path}\n\n"
            f"This term hasn't been scraped yet. To fix this:\n"
            f"  1. Get the enrollment database CSV for this term (e.g. "
            f"database.{term_code}.csv) from SFU's course offerings data.\n"
            f"  2. Run: python3 get_prereqs_and_schedule.py\n"
            f"     (update the enrollment_csv/year/term arguments in its "
            f"__main__ block to match this term first)\n"
            f"  3. Re-run this scheduler once "
            f"sfu_prereqs_and_schedule_{term_code}.csv exists.\n"
        ) from None

    df['prereq_codes'] = df['prereq_codes'].fillna('').apply(
        lambda s: [c for c in s.split(';') if c] if s else []
    )
    # schedule comes in as Python-dict-style text, not JSON, remember to use literal_eval
    df['schedule'] = df['schedule'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s else []
    )

    # older scrapes won't have this column yet - fall back to "no instructor known"
    # for every section rather than erroring, so rating-based scheduling just
    # degrades to picking any valid schedule.
    if 'instructors' in df.columns:
        df['instructors'] = df['instructors'].fillna('').apply(
            lambda s: [n.strip() for n in s.split(';') if n.strip()] if isinstance(s, str) and s else []
        )
    else:
        df['instructors'] = [[] for _ in range(len(df))]

    courses_taken = {c['course'] for c in transcript_result['courses']}

    return {
        'major': transcript_result['major'],
        'cgpa': transcript_result['cgpa'],
        'courses_taken': courses_taken,
        'course_data': df,
        'term_code': term_code,
    }


def filter_by_prerequisites(course_data, courses_taken):
    """
    Return only rows the student is currently eligible for:
      - all prereq_codes satisfied by courses_taken
      - not flagged prereq_unparseable (permission/co-op-based reqs can't be
        auto-verified in the MVS, so they're excluded rather than assumed met)
      - not already taken
    """
    def is_eligible(row):
        if row['prereq_unparseable']:
            return False
        if row['course'] in courses_taken:
            return False
        return set(row['prereq_codes']).issubset(courses_taken)

    mask = course_data.apply(is_eligible, axis=1)
    return course_data[mask].reset_index(drop=True)


def _parse_days(days_str):
    """Normalize a days string ('Mo, We, Fr' or 'Mo We Fr') into a list of day codes."""
    if not days_str:
        return []
    parts = re.split(r'[,\s]+', days_str.strip())
    return [p for p in parts if p]


def _time_to_minutes(time_str):
    """Convert 'HH:MM' 24h string to minutes since midnight."""
    h, m = time_str.split(':')
    return int(h) * 60 + int(m)


def _slots_overlap(slot_a, slot_b):
    """True if two schedule slots share a day and their time ranges overlap."""
    days_a = set(_parse_days(slot_a.get('days', '')))
    days_b = set(_parse_days(slot_b.get('days', '')))
    if not days_a & days_b:
        return False
    start_a, end_a = _time_to_minutes(slot_a['startTime']), _time_to_minutes(slot_a['endTime'])
    start_b, end_b = _time_to_minutes(slot_b['startTime']), _time_to_minutes(slot_b['endTime'])
    return start_a < end_b and start_b < end_a


def check_schedule(term_schedule):
    """
    Validate a candidate set of courses (list of course_data rows/dicts).
    Returns (is_valid, reason), reason is None if valid.
    """
    n = len(term_schedule)
    if n < 3:
        return False, f"Too few courses ({n}, need 3-5)"
    if n > 5:
        return False, f"Too many courses ({n}, need 3-5)"

    for i in range(len(term_schedule)):
        for j in range(i + 1, len(term_schedule)):
            course_a, course_b = term_schedule[i], term_schedule[j]
            for slot_a in course_a['schedule']:
                for slot_b in course_b['schedule']:
                    if _slots_overlap(slot_a, slot_b):
                        return False, (f"Time conflict: {course_a['course']} and "
                                       f"{course_b['course']} overlap on "
                                       f"{set(_parse_days(slot_a['days'])) & set(_parse_days(slot_b['days']))}")

    return True, None


def term_schedule_generator(eligible_courses, min_courses=3, max_courses=5, preset=None):
    """
    Basically the main Constraint Satisfaction Logic.

    Backtracking search over eligible_courses (a DataFrame with possibly
    multiple sections per course code). Picks at most one section per course,
    trying each section in turn and backtracking on conflict, to find a
    combination of min_courses-max_courses with no time conflicts.

    `preset` (optional) is a list of already-chosen section rows (e.g. the
    highly-rated "core" picked by `term_schedule_generator_maximize_ratings`)
    to keep and build the rest of the schedule around. Their course codes are
    excluded from the search so nothing gets picked twice.

    Returns the first valid schedule found (list of section rows as dicts),
    or None if no valid combination exists.
    """
    preset = list(preset) if preset else []
    preset_codes = {s['course'] for s in preset}

    grouped = eligible_courses.groupby('course')
    course_codes = [c for c in grouped.groups.keys() if c not in preset_codes]
    sections_by_course = {code: grouped.get_group(code).to_dict('records')
                           for code in course_codes}

    def backtrack(index, chosen):
        if len(chosen) >= min_courses:
            valid, _ = check_schedule(chosen)
            if valid:
                return list(chosen)

        if index >= len(course_codes) or len(chosen) >= max_courses:
            return None

        course_code = course_codes[index]

        for section_row in sections_by_course[course_code]:
            chosen.append(section_row)
            no_conflict = all(
                not _slots_overlap(s_a, s_b)
                for k in range(len(chosen) - 1)
                for s_a in chosen[k]['schedule']
                for s_b in chosen[-1]['schedule']
            )
            if no_conflict:
                result = backtrack(index + 1, chosen)
                if result is not None:
                    return result
            chosen.pop()

        return backtrack(index + 1, chosen)

    return backtrack(0, list(preset))


def _section_rating(section, rating_lookup):
    """Average RMP rating of a section's instructor(s), or None if none are rated."""
    ratings = [r for r in (rating_lookup(name) for name in section.get('instructors', [])) if r is not None]
    if not ratings:
        return None
    return sum(ratings) / len(ratings)


def _best_rated_subset(candidates, k):
    """
    Among `candidates` (list of (course_code, section, rating), sorted by
    rating descending), find the conflict-free subset of exactly `k` sections
    (at most one per course_code) that maximizes total rating.

    Branch-and-bound: at each candidate we either include it (if its course
    isn't already used and it doesn't conflict with what's chosen) or skip
    it. Pruned using the sum of the next `need` ratings in the (already
    sorted-descending) list as an upper bound on what's still achievable -
    that's always >= the true max since it ignores course-dupes/conflicts.
    """
    n = len(candidates)
    ratings_desc = [c[2] for c in candidates]
    best = {'schedule': None, 'sum': float('-inf')}

    def upper_bound(i, need):
        return sum(ratings_desc[i:i + need])

    def backtrack(i, chosen, used_codes, current_sum):
        need = k - len(chosen)
        if need == 0:
            if current_sum > best['sum']:
                best['sum'] = current_sum
                best['schedule'] = list(chosen)
            return
        if i >= n or (n - i) < need:
            return
        if current_sum + upper_bound(i, need) <= best['sum']:
            return

        code, section, rating = candidates[i]
        if code not in used_codes:
            conflict = any(
                _slots_overlap(s_a, s_b)
                for c in chosen for s_a in c['schedule'] for s_b in section['schedule']
            )
            if not conflict:
                chosen.append(section)
                used_codes.add(code)
                backtrack(i + 1, chosen, used_codes, current_sum + rating)
                used_codes.discard(code)
                chosen.pop()
        backtrack(i + 1, chosen, used_codes, current_sum)

    backtrack(0, [], set(), 0.0)
    return best['schedule']


def _best_rated_core(eligible_courses, rating_lookup, max_k):
    """
    Find the largest non-conflicting set (up to max_k, one section per
    course) of sections whose instructor has a known RMP rating, breaking
    ties by highest average rating. Returns [] if no section has a rated
    instructor.
    """
    grouped = eligible_courses.groupby('course')
    candidates = []
    for code, group in grouped:
        for section in group.to_dict('records'):
            rating = _section_rating(section, rating_lookup)
            if rating is not None:
                candidates.append((code, section, rating))
    candidates.sort(key=lambda c: c[2], reverse=True)

    for k in range(min(max_k, len(candidates)), 0, -1):
        result = _best_rated_subset(candidates, k)
        if result:
            return result
    return []


def term_schedule_generator_maximize_ratings(eligible_courses, rating_lookup,
                                              min_courses=3, max_courses=5):
    """
    Same CSP as term_schedule_generator, but prefers professors with strong
    RateMyProfessor ratings: it first fills as many of the min_courses-max_courses
    slots as possible with well-rated, non-conflicting sections (more rated
    professors beats a single stand-out one padded with unknowns), tie-breaking
    by highest average rating among sections of equal count. Only if that
    "rated core" doesn't reach min_courses does it pad with other eligible
    courses (rated or not) to meet the minimum.

    Falls back to the behavior of `term_schedule_generator` (first valid
    schedule) when no section's instructor is found in the RMP data - e.g.
    before the scraper has been re-run to capture instructor names.
    """
    rated_core = _best_rated_core(eligible_courses, rating_lookup, max_courses)
    return term_schedule_generator(eligible_courses, min_courses=min_courses,
                                    max_courses=max_courses, preset=rated_core)


def main(transcript_pdf, enrollment_date=None, data_dir=DEFAULT_DATA_DIR):
    """
    Full pipeline: transcript -> eligible courses -> CSP Logic -> valid schedule.

    Major is always taken from the transcript, never asked for separately.
    If enrollment_date isn't provided (as a datetime.date), the student is
    prompted for it in the command line.
    """
    print(f"Parsing transcript: {transcript_pdf}")
    transcript_result = parse_sfu_transcript(transcript_pdf)
    print(f"  Major: {transcript_result['major']}")
    print(f"  Completed courses: {len(transcript_result['courses'])}")

    if enrollment_date is None:
        while True:
            raw = input("\nWhen do you plan to enroll? (e.g. 2024-09-15): ")
            try:
                enrollment_date = parse_enrollment_date(raw)
                break
            except ValueError as e:
                print(e)

    year, term = enrollment_date_to_term(enrollment_date)
    print(f"  Target term: {term.capitalize()} {year}")

    print("Loading course data...")
    data = load_data(transcript_result, data_dir=data_dir, year=year, term=term)
    print(f"  Term code: {data['term_code']}")

    print("Filtering by prerequisites...")
    eligible = filter_by_prerequisites(data['course_data'], data['courses_taken'])
    print(f"  Eligible courses/sections: {len(eligible)}")

    if eligible.empty:
        print("No eligible courses found - cannot generate a schedule.")
        return None

    print("Loading professor ratings...")
    rating_lookup = professor_ratings.make_rating_lookup()

    print("Searching for a schedule with the best-rated professors...")
    schedule = term_schedule_generator_maximize_ratings(eligible, rating_lookup)

    if schedule is None:
        print("No valid 3-5 course schedule found with the given eligible courses.")
        return None

    print(f"\nValid schedule found ({len(schedule)} courses):")
    for section in schedule:
        print(f"  {section['course']} {section['section']}:")
        for slot in section['schedule']:
            print(f"      {slot['days']} {slot['startTime']}-{slot['endTime']} ({slot['campus']})")
        if section['instructors']:
            rating = _section_rating(section, rating_lookup)
            rating_str = f"{rating:.1f}" if rating is not None else "no RMP rating"
            print(f"      Instructor(s): {', '.join(section['instructors'])} ({rating_str})")

    return schedule


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scheduler.py <transcript.pdf>")
        print("You'll be prompted for your enrollment date.")
        sys.exit(1)

    pdf_path = sys.argv[1]
    main(pdf_path)