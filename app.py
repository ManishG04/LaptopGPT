from flask import Flask, url_for, render_template, jsonify

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    
    return jsonify({"message": "Recommendation system under development"})

if __name__ == "__main__":
    app.run(debug=True)
