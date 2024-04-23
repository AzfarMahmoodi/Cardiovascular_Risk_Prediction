# Cardiovascular_Risk_Prediction


for prediction


heart_data = pd.read_csv('heart_disease_data.csv')
heart_data['target'].value_counts()
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model=KNeighborsClassifier()
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)



X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)



#input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
#59,1,0,164,176,1,0,90,0,1,1,2,1
#62,0,0,140,268,0,0,160,0,3.6,0,2,2
# Take input from the user
age = float(input("Enter age: "))
sex = float(input("Enter sex (0 for female, 1 for male): "))
cp = float(input("Enter chest pain type (0-3): "))
trestbps = float(input("Enter resting blood pressure: "))
chol = float(input("Enter serum cholesterol: "))
fbs = float(input("Enter fasting blood sugar > 120 mg/dl (1 for true, 0 for false): "))
restecg = float(input("Enter resting electrocardiographic results (0-2): "))
thalach = float(input("Enter maximum heart rate achieved: "))
exang = float(input("Enter exercise induced angina (1 for yes, 0 for no): "))
oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
slope = float(input("Enter the slope of the peak exercise ST segment (0-2): "))
ca = float(input("Enter number of major vessels (0-3) colored by flourosopy: "))
thal = float(input("Enter thalassemia (3 for normal, 6 for fixed defect, 7 for reversable defect): "))

# Convert input into a tuple
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)



# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print("*************************************************************************")
print("\n\n")

print(prediction)

print("\n\n")

print("*************************************************************************")
print("\n\n")

if (prediction[0]== 0):
  print('The Person does not have a Cardiovascular Disease')
else:
  print('The Person has Cardiovascular Disease')
  
print("\n\n")




output:


Enter age: 67
Enter sex (0 for female, 1 for male): 0
Enter chest pain type (0-3): 2
Enter resting blood pressure: 154
Enter serum cholesterol: 233
Enter fasting blood sugar > 120 mg/dl (1 for true, 0 for false): 1
Enter resting electrocardiographic results (0-2): 2
Enter maximum heart rate achieved: 122
Enter exercise induced angina (1 for yes, 0 for no): 1
Enter ST depression induced by exercise relative to rest: 0
Enter the slope of the peak exercise ST segment (0-2): 2
Enter number of major vessels (0-3) colored by flourosopy: 2
Enter thalassemia (3 for normal, 6 for fixed defect, 7 for reversable defect): 6
*************************************************************************
[1]



*************************************************************************
The Person has Cardiovascular Disease
