from language_tool_python import LanguageTool
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained SVM model, vectorizer, and label encoder
model_path = r"C:\vignesh\sample\svm_model.pkl"
vectorizer_path = r"C:\vignesh\sample\vectorizer.pkl"
label_encoder_path = r"C:\vignesh\sample\label_encoder.pkl"
svm_model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
label_encoder = joblib.load(label_encoder_path)

@app.route('/')
def index():
    return render_template('page2.html')

@app.route('/classify', methods=['POST'])
def classify():
    input_text = request.form['input_text']  # Move this line up
    
    bad_words_df = pd.read_csv(r"C:\Users\vigne\Downloads\bad-words.csv")
    bad_words_list = bad_words_df.iloc[:, 0].tolist()
    
    # Check if any word from the CSV is present in the input text
    a=0
    for word in bad_words_list:
        if word in input_text:
            a=1
    
    # Check if input_text has more than 150 words
    word_count = len(input_text.split())
    if(a==1):
        result="The text is Written by Human"
    elif word_count < 10:
        return '''
        <script>
        alert("The text must contain at least 10 words to be classified.");
        window.location.href = "/";
        </script>
        '''
    else:
        # Check for grammatical mistakes in the input text
        tool = LanguageTool('en-US')
        matches = tool.check(input_text)
        
        if len(matches) > 0:
            result = "The text is written by a Human."
        else:
            # Preprocess the input text
            X_input = vectorizer.transform([input_text])
            
            # Make a prediction using the SVM model
            prediction = svm_model.predict(X_input)
            
            # Decode the prediction to a class label
            class_label = label_encoder.inverse_transform(prediction)
            
            # Determine the result based on the SVM model prediction
            result = f"The text is written by a {'Human' if class_label[0] == 'human' else 'AI'}"
    
    return render_template('page2.html', result=result, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)