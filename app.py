import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('C:/Users/thema/Python Projects/Research_Project/cvss_score_prediction/models/cvss_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TF-IDF vectorizer
with open('C:/Users/thema/Python Projects/Research_Project/cvss_score_prediction/models/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_summary = pickle.load(file)

# Load the OneHotEncoder
with open('C:/Users/thema/Python Projects/Research_Project/cvss_score_prediction/models/onehot_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Define the prediction function
def predict_cvss(cwe_code, access_authentication, access_complexity, access_vector, 
                 impact_availability, impact_confidentiality, impact_integrity, summary):
    # Prepare the input data for OneHotEncoder
    input_data = {
        'cwe_code': [cwe_code],
        'access_authentication': [access_authentication],
        'access_complexity': [access_complexity],
        'access_vector': [access_vector],
        'impact_availability': [impact_availability],
        'impact_confidentiality': [impact_confidentiality],
        'impact_integrity': [impact_integrity],
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # One-hot encode the categorical features
    encoded_features = encoder.transform(input_df[['access_authentication', 'access_complexity', 'access_vector',
                                                   'impact_availability', 'impact_confidentiality', 
                                                   'impact_integrity']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

    # Transform the summary text using the TF-IDF vectorizer
    summary_features = tfidf_summary.transform([summary])
    tfidf_summary_df = pd.DataFrame(summary_features.toarray(), columns=tfidf_summary.get_feature_names_out())

    # Combine the input data with the encoded features and TF-IDF features
    input_data_combined = pd.concat([input_df[['cwe_code']], encoded_df, tfidf_summary_df], axis=1)

    # Ensure the columns match the model's expected input
    input_data_combined = input_data_combined.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make the prediction
    predicted_cvss = model.predict(input_data_combined)

    return predicted_cvss[0]

# Home route to display the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the feature data from the form
        cwe_code = int(request.form['cwe_code'])
        access_authentication = request.form['access_authentication']
        access_complexity = request.form['access_complexity']
        access_vector = request.form['access_vector']
        impact_availability = request.form['impact_availability']
        impact_confidentiality = request.form['impact_confidentiality']
        impact_integrity = request.form['impact_integrity']
        summary = request.form['summary']

        # Get the predicted CVSS score
        predicted_cvss = predict_cvss(cwe_code, access_authentication, access_complexity, 
                                      access_vector, impact_availability, 
                                      impact_confidentiality, impact_integrity, summary)

        # Render the result back in the HTML
        return render_template('index.html', prediction=predicted_cvss,
                               cwe_code=cwe_code, access_authentication=access_authentication, 
                               access_complexity=access_complexity, access_vector=access_vector,
                               impact_availability=impact_availability, 
                               impact_confidentiality=impact_confidentiality,
                               impact_integrity=impact_integrity, summary=summary)
    
    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
