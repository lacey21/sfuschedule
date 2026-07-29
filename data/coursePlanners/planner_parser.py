# importing required modules
import re
import os
import pandas as pd
from pypdf import PdfReader

#Get all pdfs from current directory

coursePlanners = [f for f in os.listdir('.') if f.endswith('.pdf')]

for (pdf_file) in coursePlanners:
    # creating a pdf reader object
    reader = PdfReader(pdf_file)

    # printing number of pages in pdf file
    print(len(reader.pages))

    # getting a specific page from the pdf file
    page = reader.pages[0]

    # extracting text from page
    text = page.extract_text()

    # Check year from title
    year = pdf_file[pdf_file.find("20"):pdf_file.find("20")+4]
    sem=pdf_file[pdf_file.find("Fall"):pdf_file.find("Fall")+4] or pdf_file[pdf_file.find("Spring"):pdf_file.find("Spring")+6] or pdf_file[pdf_file.find("Summer"):pdf_file.find("Summer")+6]
    print(year)
    print(sem)
    # courseOffering="C:\\Users\\User\\OneDrive - Simon Fraser University (1sfu)\\CMPT 310-DESKTOP-0SREQNB\\sfuschedule\\data\\courseOfferings\\database.1247.csv"
    # # Check if no year is found in the title
    # if year=="2022" and sem=="Fall":
    #     courseOffering="C:\\Users\\User\\OneDrive - Simon Fraser University (1sfu)\\CMPT 310-DESKTOP-0SREQNB\\sfuschedule\\data\\courseOfferings\\database.1227.csv"
    # elif year=="2023" and sem=="Spring":
    #     courseOffering="C:\\Users\\User\\OneDrive - Simon Fraser University (1sfu)\\CMPT 310-DESKTOP-0SREQNB\\sfuschedule\\data\\courseOfferings\\database.1231.csv"
    # elif year=="2023" and sem=="Fall":
    #     courseOffering="C:\\Users\\User\\OneDrive - Simon Fraser University (1sfu)\\CMPT 310-DESKTOP-0SREQNB\\sfuschedule\\data\\courseOfferings\\database.1237.csv"
    # elif year=="2024" and sem=="Fall":
    #     courseOffering="C:\\Users\\User\\OneDrive - Simon Fraser University (1sfu)\\CMPT 310-DESKTOP-0SREQNB\\sfuschedule\\data\\courseOfferings\\database.1247.csv"
    # elif year=="2024":
    #     courseOffering="C:\\Users\\User\\OneDrive - Simon Fraser University (1sfu)\\CMPT 310-DESKTOP-0SREQNB\\sfuschedule\\data\\courseOfferings\\database.1237.csv"

    courseOffering="C:\\Users\\DELL\\OneDrive\\Desktop\\SFU\\Courses\\Summer 2026\\CMPT 310\\sfuschedule\\data\\courseOfferings\\database.1247.csv"
        # Check if no year is found in the title
    if year=="2022" and sem=="Fall":
            courseOffering="C:\\Users\\DELL\\OneDrive\\Desktop\\SFU\\Courses\\Summer 2026\\CMPT 310\\sfuschedule\\data\\courseOfferings\\database.1227.csv"
    elif year=="2023" and sem=="Spring":
            courseOffering="C:\\Users\\DELL\\OneDrive\\Desktop\\SFU\\Courses\\Summer 2026\\CMPT 310\\sfuschedule\\data\\courseOfferings\\database.1231.csv"
    elif year=="2023" and sem=="Fall":
            courseOffering="C:\\Users\\DELL\\OneDrive\\Desktop\\SFU\\Courses\\Summer 2026\\CMPT 310\\sfuschedule\\data\\courseOfferings\\database.1237.csv"
    elif year=="2024" and sem=="Fall":
            courseOffering="C:\\Users\\DELL\\OneDrive\\Desktop\\SFU\\Courses\\Summer 2026\\CMPT 310\\sfuschedule\\data\\courseOfferings\\database.1247.csv"
    elif year=="2024":
            courseOffering="C:\\Users\\DELL\\OneDrive\\Desktop\\SFU\\Courses\\Summer 2026\\CMPT 310\\sfuschedule\\data\\courseOfferings\\database.1237.csv"

    #Extract courseofferings into lists
    # df = pd.read_csv(courseOffering)
    if courseOffering.endswith('.xlsx'):
        df = pd.read_excel(courseOffering)
    else:
        df = pd.read_csv(courseOffering)
    # print(df.columns.tolist())  #comment out after checking the column names
    subject = df['Subject']
    code = df['CatNbr']
    courseCodes=subject+" "+code.astype(str)
    courseCodes=courseCodes.tolist()
    print(courseCodes[:10])
    print(len(courseCodes))

    #Extract all course codes from the text ex. ENGL 300W or CMPT 210 
    courseCodesInText = re.findall(r'[A-Z]{4} \d{3}[A-Z]?', text)
    courseCodesInText += re.findall(r'[A-Z]{4} \d{1,9}[X]{2}', text)
    courseCodesInText = list(set(courseCodesInText))
    
    #Write the course codes in the text to a file
    with open(f"{pdf_file[:-4]}_course_codes.txt", "w") as f:
        for courseCode in courseCodesInText:
            f.write(courseCode + "\n")
