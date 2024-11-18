from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the paraphrasing model
def load_paraphraser():
    try:
        print("Loading paraphrasing model...")
        paraphraser = pipeline("text2text-generation", model="t5-large")
        print("Model loaded successfully!")
        return paraphraser
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize the paraphraser
paraphraser = load_paraphraser()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    if not paraphraser:
        return jsonify({"error": "Model not loaded. Please try again later."}), 500

    # Get input text from the form
    input_text = request.form.get("input_text", "").strip()
    if not input_text:
        return jsonify({"error": "No input text provided."}), 400

    try:
        # Perform paraphrasing
        prompt = f"Rewrite the following text: {input_text}"
        results = paraphraser(
            prompt,
            max_length=200,
            num_return_sequences=1,
            num_beams=5,
            temperature=1.0,
            do_sample=True
        )
        paraphrased_text = results[0]['generated_text'].replace("Rewrite the following text:", "").strip()
        return jsonify({"paraphrased_text": paraphrased_text})
    except Exception as e:
        return jsonify({"error": f"An error occurred during paraphrasing: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
