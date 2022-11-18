# Importing the libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('student-mat.csv')
print("Data read")
print(df.shape())
# rename column labels
df.columns = ['school', 'sex', 'age', 'address', 'family_size', 'parents_status', 'mother_education', 'father_education',
              'mother_job', 'father_job', 'reason', 'guardian', 'commute_time', 'study_time', 'failures', 'school_support',
              'family_support', 'paid_classes', 'activities', 'nursery', 'desire_higher_edu', 'internet', 'romantic', 'family_quality',
              'free_time', 'go_out', 'weekday_alcohol_usage', 'weekend_alcohol_usage', 'health', 'absences', 'period1_score', 'period2_score', 'final_score']

# convert final_score to categorical variable # Good:15~20 Fair:10~14 Poor:0~9
df['final_grade'] = 'na'
df.loc[(df.final_score >= 16) & (df.final_score <= 20), 'final_grade'] = '9.0 CGPA and above'
df.loc[(df.final_score >= 11) & (df.final_score <= 15), 'final_grade'] = '8.0 CGPA to 8.9 CGPA'
df.loc[(df.final_score >= 0) & (df.final_score <= 10), 'final_grade'] = 'below 6.0 CGPA'


# create dataframe dfd for classification
final_df = df.copy()
final_df = final_df.drop(['final_score'], axis=1)

# label encode final_grade

le = preprocessing.LabelEncoder()
final_df.final_grade = le.fit_transform(final_df.final_grade)
print(le.classes_)
# dataset train_test_split

X = final_df.drop('final_grade', axis=1)
y = final_df.final_grade
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_test.iloc[0].values)
print(X_test.shape)
# get dummy varibles
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
print(X_test.iloc[0].values)
param_grid_forest = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 11, 12, 13, 14, 15, 20, 25],
    'criterion': ['gini', 'entropy']
}


rf_final = RandomForestClassifier(
    max_features='auto', n_estimators=300, max_depth=12, criterion='gini', random_state=42)
rf_final = rf_final.fit(X_train, y_train)

print("Random Forest Classifier Model Training Score", ":", rf_final.score(X_train, y_train), ",",
      "Test Score", ":", rf_final.score(X_test, y_test))


# Saving model to disk
pickle.dump(rf_final, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict(X_test))
