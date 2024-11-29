from flask import Flask, render_template, request, jsonify
from chatbot import generate_response, handle_followup_questions
import markdown2

app = Flask(__name__)

context = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        global context
        data = request.json  # Read the JSON payload
        if not data:
            return jsonify({"response": "Invalid JSON. Please provide a valid message."}), 400

        user_message = data.get('message')
        if not user_message:
            return jsonify({"response": "Message field is required."}), 400

        if context:  # If context exists, process follow-up questions
            # Pass the context and user message to the follow-up handler
            response = handle_followup_questions(user_message, context)
        else:  # If no context, generate the initial recommendation
            response, context = generate_response(user_message, context)

        # Convert response to Markdown HTML
        formatted_response = markdown2.markdown(response)

        return jsonify({"response": formatted_response})

    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Respond with no content for favicon requests

if __name__ == "__main__":
    app.run(debug=True)
