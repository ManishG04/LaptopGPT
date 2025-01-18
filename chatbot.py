import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from recommendation import filter_laptops
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

    **Use Case Mappings:**

    1. "Student/Basic Use" (Web browsing, documents, basic multitasking):
    {
        "specifications": {
            "RAM (in GB)": 8,
            "Storage": "256",
            "Screen Size (in inch)": 14.0,
            "dedicated_graphics": false
        },
        "price_range": {
            "min": 30000,
            "max": 60000
        },
        "performance_range": {
            "min": 30,
            "max": 75
        },
        "portability_range": {
            "min": 60,
            "max": 100
        }
    }

    2. "Gaming" (High-performance gaming, streaming):
    {
        "specifications": {
            "RAM (in GB)": 16,
            "Storage": "512",
            "Screen Size (in inch)": 15.6,
            "processor_min": "i5",
            "dedicated_graphics": true
        },
        "price_range": {
            "min": 80000,
            "max": 301990
        },
        "performance_range": {
            "min": 80,
            "max": 100
        },
        "portability_range": {
            "min": 0,
            "max": 70
        }
    }

    3. "Business" (Professional work, presentations, travel):
    {
        "specifications": {
            "RAM (in GB)": 16,
            "Storage": "512",
            "Screen Size (in inch)": 14.0,
            "processor_min": "i5"
        },
        "price_range": {
            "min": 60000,
            "max": 240000
        },
        "performance_range": {
            "min": 70,
            "max": 85
        },
        "portability_range": {
            "min": 50,
            "max": 100
        }
    }

    4. "Content Creation" (Video editing, 3D rendering, graphic design):
    {
        "specifications": {
            "RAM (in GB)": 32,
            "Storage": "1024",
            "Screen Size (in inch)": 15.6,
            "dedicated_graphics": true
        },
        "price_range": {
            "min": 120000,
            "max": 301990
        },
        "performance_range": {
            "min": 70,
            "max": 100
        },
        "portability_range": {
            "min": 40,
            "max": 70
        }
    }

    5. "Programming/Development" (Coding, compilation, virtual machines):
    {
        "specifications": {
            "RAM (in GB)": 16,
            "Storage": "512",
            "Screen Size (in inch)": 15.6,
            "processor_min": "i7"
        },
        "price_range": {
            "min": 70000,
            "max": 200000
        },
        "performance_range": {
            "min": 65,
            "max": 90
        },
        "portability_range": {
            "min": 40,
            "max": 100
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

    **Sample Query Mappings:**

    1. "I need a laptop for college programming":
    {
        "specifications": {
            "RAM (in GB)": 16,
            "Storage": "512",
            "Screen Size (in inch)": 15.6,
            "dedicated_graphics": false
        },
        "price_range": {
            "min": 60000,
            "max": 100000
        },
        "performance_range": {
            "min": 60,
            "max": 90
        },
        "portability_range": {
            "min": 0,
            "max": 90
        }
    }

    2. "Looking for a gaming laptop under 1 lakh":
    {
        "specifications": {
            "RAM (in GB)": 16,
            "Storage": "512",
            "Screen Size (in inch)": 15.6,
            "dedicated_graphics": true
        },
        "price_range": {
            "min": 60000,
            "max": 100000
        },
        "performance_range": {
            "min": 60,
            "max": 100
        },
        "portability_range": {
            "min": 0,
            "max": 70
        }
    }

    3. "Need a lightweight laptop for work":
    {
        "specifications": {
            "RAM (in GB)": 8,
            "Storage": "512",
            "Screen Size (in inch)": 14.0,
            "dedicated_graphics": false
        },
        "price_range": {
            "min": 45000,
            "max": 80000
        },
        "performance_range": {
            "min": 60,
            "max": 100
        },
        "portability_range": {
            "min": 60,
            "max": 100
        }
    }

    These sample queries demonstrate how to convert common user requests into structured JSON format. The specifications and ranges are optimized based on the typical requirements for each use case while considering budget constraints and portability needs.

    Always maintain this structure when parsing user queries, adjusting the values based on specific requirements mentioned in the query while using these samples as baseline references for similar requests.

    Always provide complete JSON objects with all required fields. If specific requirements aren't mentioned, use the context to infer appropriate ranges based on the use case and budget constraints. Do not give any explanation for the JSON object. Only provide the JSON object.
"""

response_rules = """
    1. Do not mention portability, value, performance and similarity scores.
    2. If the recommendations is empty or contain null values suggest generally without too much details and ask user to elaborate their needs.
    3. If the user asks for a laptop under 1 lakh, suggest a laptop under 1 lakh.
    4. If the user asks for 5 laptops suggest 5 laptops and 10 for 10 and so on. Do not suggest more than what the user asks for.
    5. If the user doesn't ask for a specific number of laptops, suggest 3 laptops.
"""


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
        recommendations = filter_laptops(parsed_input)
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
                filtered_results = filter_laptops(session["parsed_data"])
                
                if filtered_results["status"] == "success" and filtered_results["filtered_laptops"]:
                    # Format recommendations for the chatbot
                    formatted_recommendations = {
                        "best_match": filtered_results["filtered_laptops"][0],
                        "similar_recommendations": filtered_results["filtered_laptops"][1:],
                        "total_matches": filtered_results["total_matches"]
                    }
                    session["recommendations"] = formatted_recommendations
                else:
                    return "I couldn't find any laptops matching your requirements. Could you please adjust your criteria?"
                
            except json.JSONDecodeError:
                return "Could you provide more specific details about your requirements?"
        else:
            # Handle follow-up questions using existing recommendations
            formatted_recommendations = session["recommendations"]

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
        return "An error occurred while processing your request. Please try again with different requirements."