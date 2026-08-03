import random
import csv
import os

def generate_csv_data(filepath='student_performance.csv', num_records=520, seed=42):
    """Generates synthetic student performance data and saves to CSV."""
    random.seed(seed)

    genders = ['Male', 'Female']
    grades = ['A', 'B', 'C', 'D', 'F']

    data = []
    header = ['Student_ID', 'Gender', 'Age', 'Attendance', 'Study_Hours', 'Internal_Marks', 'Previous_Grade', 'Absences', 'Result']

    for i in range(1, num_records + 1):
        student_id = f"STU{1000 + i}"
        gender = random.choice(genders)
        age = random.randint(17, 24)
        
        # Study hours per week (1 to 20)
        study_hours = round(random.uniform(2.0, 16.0), 1)
        
        # Absences between 0 and 20 days
        absences = random.randint(0, 20)
        
        # Attendance % correlated with absences and random noise
        base_attendance = 100 - (absences * 3.5) + random.uniform(-4, 4)
        attendance = max(40.0, min(100.0, round(base_attendance, 1)))
        
        # Internal marks out of 50
        base_marks = 20 + (study_hours * 1.5) + (attendance * 0.15) - (absences * 0.5) + random.uniform(-3, 3)
        internal_marks = max(10, min(50, int(round(base_marks))))
        
        # Previous Grade based on internal performance
        if internal_marks >= 42:
            prev_grade = random.choice(['A', 'B'])
        elif internal_marks >= 32:
            prev_grade = random.choice(['B', 'C'])
        elif internal_marks >= 24:
            prev_grade = random.choice(['C', 'D'])
        else:
            prev_grade = random.choice(['D', 'F'])
            
        # Introduce small missing values (~2% chance) for ETL data cleaning demonstration
        study_hours_val = study_hours if random.random() > 0.02 else ""
        attendance_val = attendance if random.random() > 0.02 else ""
        internal_marks_val = internal_marks if random.random() > 0.02 else ""
        
        # Calculate score to derive ground truth target Result
        score = (study_hours * 2.5) + (attendance * 0.35) + (internal_marks * 0.75) - (absences * 1.2)
        result = "Pass" if score >= 52 else "Fail"
        
        data.append([
            student_id,
            gender,
            age,
            attendance_val,
            study_hours_val,
            internal_marks_val,
            prev_grade,
            absences,
            result
        ])

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Generated {filepath} with {num_records} records successfully.")
    return filepath

if __name__ == '__main__':
    generate_csv_data()

