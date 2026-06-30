import re
import sys
import csv
from pathlib import Path

def parse_sfu_transcript(pdf_path, output_csv=None):
    """Parse SFU unofficial transcript PDF.
    Optionally exports courses to CSV."""
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not installed. Install with: pip install pdfplumber")
        sys.exit(1)
    
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    
    # Normalize whitespace
    full_text = re.sub(r'\s+', ' ', full_text)
    
    # Extract major
    major_match = re.search(r'Major in (.*?), Bachelor of Science', full_text)
    major = major_match.group(1).strip() if major_match else "Unknown"
    
    # Extract current CGPA
    cgpa_matches = re.findall(r'Cumulative GPA:\s*([\d.]+)', full_text)
    cgpa = float(cgpa_matches[-1]) if cgpa_matches else None
    
    # Extract completed courses
    course_pattern = re.compile(
        r'([A-Z]{2,4}\s+\d+[A-Z]?)'                    # Course code
        r'\s+([A-Za-z][^0-9]+?)(?=\s+\d+\.\d{2})'      # Description
        r'\s+\d+\.\d{2}\s+(\d+\.\d{2})\s+([A-F][+-]?)', # Units + Grade
        re.IGNORECASE
    )
    
    courses = []
    seen = set()
    for match in course_pattern.finditer(full_text):
        code, desc, completed, grade = match.groups()
        code = code.strip()
        desc = desc.strip()
        if float(completed) > 0 and code not in seen:
            seen.add(code)
            courses.append({
                'course': code,
                'description': desc,
                'grade': grade
            })
    
    result = {'major': major, 'cgpa': cgpa, 'courses': courses}
    
    # Export to CSV
    if output_csv and courses:
        csv_path = Path(output_csv)
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['course', 'description', 'grade'])
                writer.writeheader()
                writer.writerows(courses)
            print(f"Courses exported to: {csv_path.absolute()}")
        except Exception as e:
            print(f"Failed to write CSV: {e}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_transcript.py <transcript.pdf> [output.csv]")
        print("Example: python parse_transcript.py mytranscript.pdf courses.csv")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = parse_sfu_transcript(pdf_file, output_csv)
    print("\n=== Transcript Summary ===")
    print("Major:", result['major'])
    print("Current CGPA:", result['cgpa'])
    print(f"\nCompleted Courses: {len(result['courses'])}")
    for course in result['courses']:
        print(f"{course['course']}: {course['description']} - Grade: {course['grade']}")