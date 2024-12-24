import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from recommendation import get_recommendations  
from pydantic import SecretStr

load_dotenv(override=True)

# Model
api_key = SecretStr(os.getenv("OPENAI_API_KEY"))
if not api_key:
    raise Exception("OpenAI API key not set")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    api_key=api_key,
)

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

knowledge_base = """
    You are an advanced laptop recommendation expert with deep knowledge of computer hardware and user requirements. Your task is to analyze user queries and convert them into a structured JSON format that captures both explicit requirements and implicit needs based on use cases.

    **Core Responsibilities:**
    1. Convert user requirements into specific technical specifications
    2. Infer appropriate performance and portability ranges based on use cases
    3. Set reasonable price ranges based on requirements
    4. Handle both technical and non-technical user queries
    5. We have Max RAM size of 32 in dataset, Storage of 1024.
    6. Screen Size we have are 13.3, 14.0, 15.6, 16.1, 17.3
    7. For Desktop replacement strictly keep portability upper range till 50
    8. The price in the dataset ranges from 15990 to 301990, if lower limit is provided set upper limit the highest and vice versa.

    **Output Format:**
    {
        "specifications": {
            "RAM (in GB)": <number>,
            "Storage": "<number>",
            "Screen Size (in inch)": <number>
        },
        "price_range": {
            "min": <number>,
            "max": <number>
        },
        "performance_range": {
            "min": <number>,
            "max": <number>
        },
        "portability_range": {
            "min": <number>,
            "max": <number>
        }
    }

    **Performance Range Guidelines:**
    - Basic/Student (web browsing, documents): 50-70
    - Productivity (multiple applications, light editing): 65-80
    - Creative Work (video editing, design): 75-90
    - Gaming/Professional: 85-100

    **Portability Range Guidelines:**
    - Desktop Replacement (>2.5kg): 0-30
    - All-Purpose (2-2.5kg): 30-60
    - Ultraportable (<2kg): 60-100

    **Example Mappings:**

    1. "I need a laptop for college, mainly for programming and light gaming"
    {
        "specifications": {
            "RAM (in GB)": 16,
            "Storage": "512",
            "Screen Size (in inch)": 15.6
        },
        "price_range": {
            "min": 60000,
            "max": 100000
        },
        "performance_range": {
            "min": 50,
            "max": 85
        },
        "portability_range": {
            "min": 40,
            "max": 80
        }
    }

    2. "Looking for a powerful workstation for video editing, budget up to 2 lakhs"
    {
        "specifications": {
            "RAM (in GB)": 32,
            "Storage": "1024",
            "Screen Size (in inch)": 15.6
        },
        "price_range": {
            "min": 100000,
            "max": 200000
        },
        "performance_range": {
            "min": 75,
            "max": 100
        },
        "portability_range": {
            "min": 0,
            "max": 50
        }
    }

    Always provide complete JSON objects with all required fields. If specific requirements aren't mentioned, use the context to infer appropriate ranges based on the use case and budget constraints.
"""

response_rules = """
    1.  If recommendation has alternatives same as the best match, do not show the best match and mention this is the only laptop you could find in your dataset.
    2. Do not mention portability, value, performance and similarity scores.
    3. If the recommendations is empty or contain null values suggest generally without too much details and ask user to elaborate their needs.
"""

# Define prompt templates
parse_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name='history'),
        ('human', "Given this knowledge base:\n{knowledge_base}\nResponde to this:\n{question}")
    ]
)

response_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name='history'),
        ('human', "Using recommendations: {recommendations}, answer the user's query: {question}. Considering these {response_rules}")
    ]
)

output_parser = StrOutputParser()

# Session memory management
context = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    """
    Get chat history by session ID.
    """
    if session_id not in context:
        context[session_id] = InMemoryChatMessageHistory()
    return context[session_id]

# Creating chains
parse_chain_with_history = RunnableWithMessageHistory(
    parse_prompt | model | output_parser,
    get_by_session_id,
    input_messages_key='question',
    history_messages_key='history',
)

response_chain_with_history = RunnableWithMessageHistory(
    response_prompt | model | output_parser,
    get_by_session_id,
    input_messages_key='question',
    history_messages_key='history',
)


def find_recommendations(parsed_input):
    try:
        recommendations = get_recommendations(parsed_input)
        if "error" in recommendations:
            raise ValueError(recommendations["error"])
        return recommendations
        
    except Exception as e:
        return {"error": f"Recommendation error: {str(e)}"}

session_state = {}

def generate_response(user_message, session_id='default'):
    try:
        if session_id not in session_state:
            session_state[session_id] = {
                "parsed_data": None,
                "recommendations": None,
            }

        session = session_state[session_id]

        if session["parsed_data"] is None:
            parsed_input = parse_chain_with_history.invoke(
                {'question': user_message, 'knowledge_base': knowledge_base},
                config={'configurable': {'session_id': session_id}}
            )
            parsed_input = strip_backticks(parsed_input)
            print("DEBUG: PARSED INPUT: ", parsed_input)
            
            try:
                session["parsed_data"] = json.loads(parsed_input)
                session["recommendations"] = find_recommendations(session["parsed_data"])
                
                formatted_recommendations = {
                    "best_match": {
                        "name": session["recommendations"]["best_match"]["name"],
                        "price": f"₹{session['recommendations']['best_match']['price']:,}",
                        "key_features": f"{session['recommendations']['best_match']['specifications']['RAM']}GB RAM, {session['recommendations']['best_match']['specifications']['Storage']}GB storage",
                        "scores": f"Performance: {session['recommendations']['best_match']['scores']['performance']}, Portability: {session['recommendations']['best_match']['scores']['portability']}"
                    },
                    "similar_recommendations": [
                        {
                            "name": rec["name"],
                            "price": f"₹{rec['price']:,}",
                            "key_differences": f"Different in: {rec['specifications']['Storage']}GB storage, {rec['specifications']['RAM']}GB RAM"
                        }
                        for rec in session["recommendations"]["similar_recommendations"]
                    ]
                }
                print("DEBUG: RECOMMENDATION: ", formatted_recommendations)
                
            except json.JSONDecodeError:
                return "Could you provide more specific details about your requirements?"
        else:
            # Handle follow-up
            formatted_recommendations = {
                "best_match": {
                    "name": session["recommendations"]["best_match"]["name"],
                    "price": f"₹{session['recommendations']['best_match']['price']:,}",
                    "key_features": f"{session['recommendations']['best_match']['specifications']['RAM']}GB RAM, {session['recommendations']['best_match']['specifications']['Storage']}GB storage",
                    "scores": f"Performance: {session['recommendations']['best_match']['scores']['performance']}, Portability: {session['recommendations']['best_match']['scores']['portability']}"
                },
                "similar_recommendations": [
                    {
                        "name": rec["name"],
                        "price": f"₹{rec['price']:,}",
                        "key_differences": f"Different in: {rec['specifications']['Storage']}GB storage, {rec['specifications']['RAM']}GB RAM"
                    }
                    for rec in session["recommendations"]["similar_recommendations"]
                ]
            }

        response = response_chain_with_history.invoke(
            {
                'question': user_message,
                'recommendations': formatted_recommendations,
                'response_rules': response_rules
            },
            config={'configurable': {'session_id': session_id}}
        )
        print("DEBUG: RESPONSE: ", response)
        return response.strip()

    except Exception as e:
        print("Error in generating response:", e)
        return "An error occurred while processing your request."