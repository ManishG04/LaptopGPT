from flask import Flask, render_template, request, jsonify
from chatbot import generate_response
import markdown2

app = Flask(__name__)

STATIC_SESSION_ID = 'test-session'

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'Invalid request. Provide a "message" field.'}), 400

        user_message = data['message']

        response = generate_response(user_message, session_id=STATIC_SESSION_ID)
        formatted_response = markdown2.markdown(response)
        
        return jsonify({"response": formatted_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/favicon.ico')
def favicon():
    return '', 204  

if __name__ == "__main__":
    app.run(debug=True)
