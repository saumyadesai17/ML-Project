from flask import Flask, render_template, request, jsonify
import urllib.parse
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('adaboost.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract the input values from the form
            url_length = float(request.form['url_length'])
            n_dots = float(request.form['n_dots'])
            n_hypens = float(request.form['n_hypens'])
            n_underline = float(request.form['n_underline'])
            n_slash = float(request.form['n_slash'])
            n_questionmark = float(request.form['n_questionmark'])
            n_at = float(request.form['n_at'])
            n_and = float(request.form['n_and'])
            # n_exclamation = float(request.form['n_exclamation'])
            # n_space = float(request.form['n_space'])
            # n_tilde = float(request.form['n_tilde'])
            # n_comma = float(request.form['n_comma'])
            # n_plus = float(request.form['n_plus'])
            # n_asterisk = float(request.form['n_asterisk'])
            # n_hastag = float(request.form['n_hastag'])
            # n_dollar = float(request.form['n_dollar'])
            # n_percent = float(request.form['n_percent'])
            # n_redirection = float(request.form['n_redirection'])

            # Create a DataFrame with the input values
            data = pd.DataFrame({
                'url_length': [url_length],
                'n_dots': [n_dots],
                'n_hypens': [n_hypens],
                'n_underline': [n_underline],
                'n_slash': [n_slash],
                'n_questionmark': [n_questionmark],
                'n_at': [n_at],
                'n_and': [n_and]
                # 'n_exclamation': [n_exclamation],
                # 'n_space': [n_space],
                # 'n_tilde': [n_tilde],
                # 'n_comma': [n_comma],
                # 'n_plus': [n_plus],
                # 'n_asterisk': [n_asterisk],
                # 'n_hastag': [n_hastag],
                # 'n_dollar': [n_dollar],
                # 'n_percent': [n_percent],
                # 'n_redirection': [n_redirection]
            })

            # Make predictions using the loaded model
            prediction = model.predict(data)

            # Convert prediction to human-readable format
            prediction_text = "Phishing Website Detected" if prediction == 1 else "Legitimate Website"

            # Render the result.html template and pass the prediction value
            return render_template('result.html', prediction=prediction_text)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return print(error_message=error_message)

# Define the route to analyze the URL and extract characteristics
@app.route('/analyze-url', methods=['GET'])
def analyze_url():
    try:
        # Get URL from request parameters
        url = request.args.get('url')

        # Decode the URL if it's URL encoded
        url = urllib.parse.unquote(url)

        # Perform URL analysis and extract characteristics
        analyzed_data = {
            'url_length': len(url),
            'n_dots': url.count('.'),
            'n_hypens': url.count('-'),
            'n_underline': url.count('_'),
            'n_slash': url.count('/'),
            'n_questionmark': url.count('?'),
            'n_at': url.count('@'),
            'n_and': url.count('&')
            # 'n_exclamation': url.count('!'),
            # 'n_space': url.count(' '),
            # 'n_tilde': url.count('~'),
            # 'n_comma': url.count(','),
            # 'n_plus': url.count('+'),
            # 'n_asterisk': url.count('*'),
            # 'n_hashtag': url.count('#'),
            # 'n_dollar': url.count('$'),
            # 'n_percent': url.count('%'),
            # 'n_redirection': 0  # Assuming redirection count is not available in the URL
        }

        # Return the analyzed data as JSON
        return jsonify(analyzed_data)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
