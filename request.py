import requests
import pandas as pd


url = 'http://localhost:5000/predict_api'

cols = ['school', 'sex', 'age', 'address', 'family_size', 'parents_status', 'mother_education', 'father_education',
        'mother_job', 'father_job', 'reason', 'guardian', 'commute_time', 'study_time', 'failures', 'school_support',
        'family_support', 'paid_classes', 'activities', 'nursery', 'desire_higher_edu', 'internet', 'romantic', 'family_quality',
        'free_time', 'go_out', 'weekday_alcohol_usage', 'weekend_alcohol_usage', 'health', 'absences', 'period1_score', 'period2_score']

values = ['GP', 'M', 18, 'R', 'GT3', 'T', 3, 2, 'other', 'other', 'course', 'mother', 1, 3, 0,
          'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 5, 3, 2, 1, 1, 3, 1, 13, 12]

df = pd.DataFrame(values, columns=cols)
print(df.shape)
newdf = pd.get_dummies(df)
print(newdf.shape)

dummycols = ['age', 'mother_education', 'father_education', 'commute_time',
             'study_time', 'failures', 'family_quality', 'free_time', 'go_out',
             'weekday_alcohol_usage', 'weekend_alcohol_usage', 'health', 'absences',
             'period1_score', 'period2_score', 'school_GP', 'school_MS', 'sex_F',
             'sex_M', 'address_R', 'address_U', 'family_size_GT3', 'family_size_LE3',
             'parents_status_A', 'parents_status_T', 'mother_job_at_home',
             'mother_job_health', 'mother_job_other', 'mother_job_services',
             'mother_job_teacher', 'father_job_at_home', 'father_job_health',
             'father_job_other', 'father_job_services', 'father_job_teacher',
             'reason_course', 'reason_home', 'reason_other', 'reason_reputation',
             'guardian_father', 'guardian_mother', 'guardian_other',
             'school_support_no', 'school_support_yes', 'family_support_no',
             'family_support_yes', 'paid_classes_no', 'paid_classes_yes',
             'activities_no', 'activities_yes', 'nursery_no', 'nursery_yes',
             'desire_higher_edu_no', 'desire_higher_edu_yes', 'internet_no',
             'internet_yes', 'romantic_no', 'romantic_yes']

dummyvals = [16, 4, 3, 2, 1, 0, 3, 3, 3, 1, 1,  4, 2, 11, 15, 1, 0, 0, 1, 1, 0, 1, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 1, 0,  0,  0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1,  1, 0,
             0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

input = dict(zip(dummycols, dummyvals))

res = requests.post(url, json=input)

print(res.json())
