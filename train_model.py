import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#Mounting the dataset
csv_file_path = './Crop_recommendation.csv'
df = pd.read_csv(csv_file_path)

#Cleaning the data
le = LabelEncoder()
X = df.drop("label", axis=1)
Y = df.label
Y = le.fit_transform(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=101)

#Training the model
devhero = RandomForestClassifier(n_estimators=100, random_state=101)
result = devhero.fit(x_train, y_train)
#y_pred = devhero.predict(x_test)



# Save the model
model_filename = './Crop_recommendation.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump({"model": devhero, "encoder": le}, file)

print(f"Model saved to {model_filename}")