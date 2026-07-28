# AI SFU Schedule Helper 
Our projects takes in your transcript and enrollment date to help generate a complete course schedule for you for the upcoming term, it considers your major to determine required courses, maximizes professor ratings, ensure enrollment success, and balances course difficulty. 
## Repository Layout

| Path | Purpose |
|------|---------|
| `code/` | Python code to run the models & parser files|
| `code/scheduler.py` | Main application to run the CSP |
| `code/decisionTreeModel.py` | Train/validate class fullness decision tree |
| `code/decisionTree.py` | Quick prediction for specific course |
| `data/` | CSVs/Excel files of Data for the model |
| `data/courseFillByWeek/` | Historical data of class fullness each semester |
| `data/courseOfferings/` | Database of the courses offered each semester |
| `data/coursePlanners/` | Required courses for each program/concentration |
| `data/sfu_rmp/` | Rate my Professors ratings for SFU courses/profs |
| `data/synthetic transcript/` | Generate transcripts for testing |

*There are additional files to create different modules and parsers 
## Required Modules
To run the scheduler, decision tree, and other helper files ensure you've installed:
```pip install ast re sys os datetime pandas openpyxl scikit-learn joblib requests pdfplumber```

## Running the CSP
```
  cd code
  python scheduler.py <transcript.pdf>
```
*To test you can use the transcripts under `data/synthetic transcript/`



