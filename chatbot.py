import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from recommendation import find_best_recommendation  
# Load environment variables
load_dotenv()

# Initialize the OpenAI model
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise Exception("OpenAI API key not set")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    api_key=api_key,
)

context = {}

def strip_backticks(code):
    """
    Remove backticks and code fences from a string.
    """
    if code.startswith("```") and code.endswith("```"):
        code = code[3:]
        first_newline_index = code.find("\n")
        if first_newline_index != -1:
            code = code[first_newline_index + 1:]
        code = code.rstrip("`")
    return code

# Define prompt templates
parse_prompt = ChatPromptTemplate.from_template(
    """
    You are helping with laptop recommendations. Use these mappings:
    - Laptop Type:
        Gaming laptop → 1
        Thin and light → 2
        2 in 1 → 3
        Notebook → 4
        Laptop → 5
        2 in 1 gaming → 6
        Business → 7
        Chromebook → 8
        Creator → 9

    Convert this user input into a structured JSON object with numeric values for laptop type and other specs. Do not provide any explanations; only return the JSON object:
    "{user_input}"

    Example output:
    {{
        "Type": 1,
        "Screen Size": 15.6,
        "Price Range": {{"min": 50000, "max": 70000}},
        "RAM": 16,
        "SSD": 512
    }}
    """
)

response_prompt = ChatPromptTemplate.from_template(
    """
    Based on the user's requirements:
    {parsed_input}

    Provide a recommendation for a suitable laptop. Ensure your response is in JSON format.
    """
)

output_parser = StrOutputParser()

def parse_user_input(user_input):
    """
    Parse the user's input to extract key laptop specifications.
    """
    print("DEBUG: Parsing message: ", user_input)
    chain = parse_prompt | model | output_parser
    try:
        response = chain.invoke({"user_input": user_input})
        print("DEBUG: Response: ", response)
        response = strip_backticks(response)
        return json.loads(response)  # Parse the JSON response
    except json.JSONDecodeError:
        return {"error": "Could not parse the response into JSON."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

def generate_response(user_message, context):
    """
    Generate a response for the user's request. Update context with laptop details.
    """
    print("DEBUG: User message: ", user_message)
    parsed_input = parse_user_input(user_message)
    if "error" in parsed_input:
        return parsed_input["error"], context

    recommendations = find_best_recommendation(parsed_input)
    if "error" in recommendations:
        return recommendations["error"], context

    # Extract recommendation details
    context.update({
        "name": recommendations.get("name", "N/A"),
        "RAM": recommendations.get("RAM (in GB)", "unknown"),
        "Storage": recommendations.get("Storage", "unknown"),
        "Price": recommendations.get("Price (in Indian Rupees)", "unknown"),
        "GPU": recommendations.get("gpu name ", "unknown"),
        "Processor": recommendations.get("Processor name", "unknown"),
        "Screen Size": recommendations.get("Screen Size (in inch)", "unknown"),
        "Operating System": recommendations.get("Operating System", "unknown"),
        "User Rating": recommendations.get("user rating", "unknown"),
    })

    # Build the response
    response = (
        f"We recommend the {context['name']}. It comes with {context['RAM']}GB RAM, "
        f"{context['Storage']}GB SSD, and is priced around ₹{context['Price']}. "
        f"It features the {context['Processor']} processor, {context['GPU']} GPU, and a "
        f"{context['Screen Size']}-inch screen. The laptop has an average user rating of {context['User Rating']}."
    )
    print("DEBUG: Generated Response:", response)
    return response, context


def handle_followup_questions(user_message, context):
    """
    Handle follow-up questions using GPT and LangChain.
    """
    followup_prompt = ChatPromptTemplate.from_template(
        """
        You are assisting with follow-up questions based on the provided laptop context.

        Context: {context}

        User Question: "{user_message}"

        Provide a clear and concise response to the user's question. If you need more words, extend appropriately.
        """
    )

    chain = followup_prompt | model | output_parser

    try:
        # Run the chain and return the response
        response = chain.invoke({"user_message": user_message, "context": context})
        print("DEBUG: GPT Follow-up Response:", response)
        return response.strip()  # Return plain text response
    except Exception as e:
        print("DEBUG: Error in Follow-up:", str(e))
        return "An error occurred while processing your follow-up question."
