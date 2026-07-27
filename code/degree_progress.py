#print("degree_progress.py started") for aakriti's testing
def load_degree_requirements(planner_file):
    required_courses = set()

    with open(planner_file, "r") as file:
        for line in file:
            course = line.strip()

            if course:
                required_courses.add(course)

    return required_courses

def extract_completed_courses(transcript_result): #getting the courses from parse transcript.py and storing them in a set
    courses_taken = set()

    for course in transcript_result['courses']:
        courses_taken.add(course['course'])

    return courses_taken

def calculate_degree_progress(courses_taken, required_courses):
    completed_courses = set() #creating a set to store completed courses

    for course in courses_taken:
        if course in required_courses:
            completed_courses.add(course) #adding the course to the completed_courses set if it is in the required_courses set

    credits_completed = len(completed_courses) * 3
    total_degree_credits = 120

    credits_remaining = total_degree_credits - credits_completed

    completion_percentage = (credits_completed / total_degree_credits) * 100

    return {
        "completed_courses": completed_courses,
        "credits_completed": credits_completed,
        "credits_remaining": credits_remaining,
        "completion_percentage": completion_percentage
    }
#again for testing purposes, you can uncomment the following code to run the script directly and see the output. Make sure to provide the correct paths to your planner file and transcript PDF.
#from parse_transcript import parse_sfu_transcript

#for testing purposes, you can uncomment the following code to run the script directly and see the output. Make sure to provide the correct paths to your planner file and transcript PDF.
#if __name__ == "__main__":

    # Load major requirements
   # required_courses = load_degree_requirements(
  #      "data/coursePlanners/ENSC PLANNER.txt"
   # )

    # Read transcript PDF
   # transcript = parse_sfu_transcript(
   #     "data/synthetic transcript/transcript_engsci_biomed.pdf"
   # )

    # Extract courses
   # student_courses = extract_completed_courses(transcript)

    # Calculate progress
  #  progress = calculate_degree_progress(
  #      student_courses,
   #     required_courses
   # )

   # print(progress)
