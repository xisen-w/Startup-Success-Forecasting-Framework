from flask import Flask, render_template, request, jsonify
import os

# Import the function from other places
from overallPipeline import mainPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def format_for_html(text):
    """
    Converts newline characters in text to HTML line breaks for display.
    """
    return text.replace("\n", "<br>")


@app.route('/analyze', methods=['POST'])
def analyze():
    startup_description = request.form['startup_description']
    founder_description = request.form['founder_description']
    mode = request.form['mode']

    # Call your mainPipeline function to perform the analysis
    analysis_result = mainPipeline(startup_description, founder_description, mode, "gpt-4")

    # Format the analysis results for HTML
    if analysis_result:
        for key, value in analysis_result.items():
            if isinstance(value, str):  # Ensure the value is text
                analysis_result[key] = format_for_html(value)

    # Direct to different templates based on the mode of analysis
    if mode == "simple":
        return render_template('simple.html', result=analysis_result)
    elif mode == "advanced":
        return render_template('advanced.html', result=analysis_result)
    else:
        return "Error: Invalid mode selected", 400

if __name__ == '__main__':
    app.run(debug=True)
