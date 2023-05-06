import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import cv2

# Load the dataset
data = pd.read_csv('drug_deaths.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Drug Death', axis=1), 
                                                    data['Drug Death'], 
                                                    test_size=0.3, 
                                                    random_state=42)

# Create a Random Forest Classifier model
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rfc.fit(X_train, y_train)

# Predict the test set labels
y_pred = rfc.predict(X_test)

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)


#Pre-Processing


# Separate the features and target variable
X = data.drop('Target_Variable', axis=1)
y = data['Target_Variable']

# Identify the categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create transformers for categorical and numerical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', LabelEncoder())
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Apply the transformers to the features
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

X = preprocessor.fit_transform(X)

#Transformaing Image
# Load the image
img = cv2.imread('your_image.jpg')

# Resize the image
resized_img = cv2.resize(img, (new_width, new_height))

# Convert the image to grayscale
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image
blurred_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)

# Apply thresholding to the image
_, thresh_img = cv2.threshold(blurred_img, threshold_value, 255, cv2.THRESH_BINARY)

# Save the transformed image
cv2.imwrite('transformed_image.jpg', thresh_img)

#Detecting Cars

# Load the image
img = cv2.imread('your_image.jpg')

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the car classifier
car_classifier = cv2.CascadeClassifier('path_to_car_classifier.xml')

# Detect cars in the image
cars = car_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the detected cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show the image with detected cars
cv2.imshow('Cars Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Face Recognizer


# Load the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained face recognition model
face_recognizer.read('path_to_trained_model.xml')

# Load the face classifier
face_classifier = cv2.CascadeClassifier('path_to_face_classifier.xml')

# Initialize the video capture device
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video feed
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Recognize faces in the frame
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_roi = gray_frame[y:y+h, x:x+w]

        # Recognize the face
        face_id, confidence = face_recognizer.predict(face_roi)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Write the recognized face label and confidence level on the image
        cv2.putText(frame, 'Face ID: {}'.format(face_id), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'Confidence: {:.2f}'.format(confidence), (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Face Recognition', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and destroy all windows
video_capture.release()
cv2.destroyAllWindows()

# Load the medical data
data = pd.read_csv('path_to_medical_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['cause_of_death'], test_size=0.2, random_state=42)

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer(max_features=10000)

# Fit the vectorizer to the training data and transform the training and testing sets
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train an SVM model on the training data
svm = SVC(kernel='linear')
svm.fit(X_train_vect, y_train)

# Predict the cause of death on the testing data
y_pred = svm.predict(X_test_vect)

# Print the classification report
print(classification_report(y_test, y_pred))
