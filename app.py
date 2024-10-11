from flask import Flask, url_for, render_template, jsonify,request

app = Flask(__name__)

laptops = [
    {"name": "Laptop A", "price": 50000, "purpose": "work"},
    {"name": "Laptop B", "price": 60000, "purpose": "gaming"}
]

@app.route('/')

def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    budget = int(request.form.get('budget'))
    purpose = request.form.get('purpose')
    recommended_laptops = [laptop for laptop in laptops if laptop['price'] <= budget and laptop['purpose'] == purpose]
    return jsonify({"message": f"Recommended Laptops: {recommended_laptops}"})


if __name__ == "__main__":
    app.run(debug=True)
