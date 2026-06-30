import requests
import pandas as pd
import time

BASE_URL = "http://www.sfu.ca/bin/wcm/course-outlines"
YEAR = "2021"
TERM = "spring"

def get_prerequisites():
    data_rows = []
    
    # 1. Get Departments
    try:
        response = requests.get(f"{BASE_URL}?{YEAR}/{TERM}")
        depts = response.json()
    except Exception as e:
        print(f"Failed to fetch departments: {e}")
        return pd.DataFrame()

    for dept in depts:
        # Some endpoints might return strings, handle that
        dept_code = dept['value'] if isinstance(dept, dict) else dept
        print(f"Processing department: {dept_code}")
        
        # 2. Get Courses in Dept
        try:
            courses = requests.get(f"{BASE_URL}?{YEAR}/{TERM}/{dept_code}").json()
        except:
            continue
        
        for course in courses:
            # FIX: Handle if course is a dict or a string
            course_num = course['value'] if isinstance(course, dict) else course
            
            # 3. Get Sections
            sections_url = f"{BASE_URL}?{YEAR}/{TERM}/{dept_code}/{course_num}"
            try:
                sections = requests.get(sections_url).json()
            except:
                continue
            
            if isinstance(sections, list) and len(sections) > 0:
                sec_name = sections[0]['value'] if isinstance(sections[0], dict) else sections[0]
                
                # 4. Get Details
                try:
                    details = requests.get(f"{BASE_URL}?{YEAR}/{TERM}/{dept_code}/{course_num}/{sec_name}").json()
                    
                    # Some details might be missing 'info', handle safely
                    info = details.get('info', {})
                    prereqs = info.get('prerequisites', 'N/A')
                    
                    data_rows.append({
                        "course": f"{dept_code} {course_num}",
                        "term": TERM,
                        "year": YEAR,
                        "prereqs": prereqs
                    })
                except:
                    continue
            
            time.sleep(0.05) 
            
    return pd.DataFrame(data_rows)

df = get_prerequisites()
df.to_csv("sfu_prerequisites_2021.csv", index=False)
print("Done! File saved as sfu_prerequisites_2021.csv")