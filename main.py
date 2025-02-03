Creating an intelligent tool for automating and optimizing invoice processing using OCR (Optical Character Recognition) and machine learning can be quite complex. Below is a simplified version of how such a tool might be structured in Python. The program will use libraries such as `pytesseract` for OCR functionalities, `pandas` for data handling, and `scikit-learn` for machine learning tasks. Note that this example assumes you have some labeled data for training a simple classifier. It also uses standard error handling practices.

Before running the program, ensure you have the required packages installed:

```bash
pip install pytesseract opencv-python pandas scikit-learn
```

Also, make sure to install Tesseract OCR on your machine. You can find instructions for installation [here](https://github.com/tesseract-ocr/tesseract).

Here's a Python program for the project:

```python
import os
import cv2
import pytesseract
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pytesseract import Output

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'YOUR_TESSERACT_PATH_HERE' # Update this path

# Function to extract text from image using pytesseract
def extract_text_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"No such file: '{image_path}'")
        text = pytesseract.image_to_string(img, output_type=Output.STRING)
        return text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None

# Function to load dataset and preprocess
def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)  # Assumes CSV format for labeled data
        if data.empty:
            raise ValueError("The dataset is empty")
        # Example preprocessing steps
        data.fillna('', inplace=True)
        # More preprocessing as required
        return data
    except FileNotFoundError:
        print(f"Data file not found: {file_path}")
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
    return None

# Function to train a machine learning model
def train_model(data):
    try:
        # Assume data has 'text' column for invoice content and 'label' column for classification
        X = data['text']
        y = data['label']
        # Simple feature extraction, convert text to a vector (could use TF-IDF, etc.)
        X_vectorized = X.apply(len).values.reshape(-1, 1)  # Placeholder feature for simplicity
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
        
        # Using RandomForest for classification
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        predictions = model.predict(X_test)
        print("Classification report:\n", classification_report(y_test, predictions))
        print("Accuracy:", accuracy_score(y_test, predictions))
        
        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

# Main processing function
def process_invoice(image_path, model):
    text = extract_text_from_image(image_path)
    if text:
        # Placeholder for feature extraction process
        feature_vector = np.array([len(text)]).reshape(1, -1)  # Simplified feature example
        try:
            prediction = model.predict(feature_vector)
            print(f"Predicted class for the invoice: {prediction}")
        except Exception as e:
            print(f"Error while making prediction: {e}")
    else:
        print("No text extracted, can't process invoice.")

# Example usage
if __name__ == "__main__":
    # Path to invoice image
    invoice_image_path = "path/to/invoice/image.jpg"
    
    # Path to labeled data
    labeled_data_path = "path/to/labeled/invoice_data.csv"
    
    # Load and preprocess labeled data
    labeled_data = load_and_preprocess_data(labeled_data_path)
    
    # Train machine learning model
    if labeled_data is not None:
        model = train_model(labeled_data)
        if model:
            # Process an example invoice
            process_invoice(invoice_image_path, model)
```

### Key Points:

1. **OCR with pytesseract**: This program uses `pytesseract` to extract text from images.
2. **Machine Learning with Scikit-learn**: It uses a simple `RandomForestClassifier` to classify invoices. Actual implementation would require more sophisticated feature extraction (like TF-IDF) for meaningful results.
3. **Error Handling**: The program includes basic error handling for file operations and model operations.
4. **Data Assumptions**: Assumes the existence of a labeled dataset in CSV format with 'text' and 'label' columns.
5. **Simplifications**: The program is simplified for demonstration. In practical applications, you would need to handle various edge cases in image reading and text preprocessing.

Ensure that you replace placeholder paths and implement actual feature extraction techniques according to your project's needs.