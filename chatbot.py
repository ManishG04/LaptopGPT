import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from recommendation import find_best_recommendation  
from pydantic import SecretStr

# Load environment variables
load_dotenv(override=True)

# Initialize the OpenAI model
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
    You are an advanced laptop recommendation expert. Your task is to recommend or convert specifications into a structured and tokenized JSON object that is highly optimized for data processing and recommendation systems. Use the following keys and tokenized values for consistent and robust outputs.

    - When a user specifies a use case, intelligently infer the required specifications and return the best possible configuration in JSON format.
    - If specs are provided, convert them into a structured JSON object using the tokenized values below, filling in missing details as needed. Ensure that every output has meaningful and complete data.
    - For missing budgets or specifications, infer values intelligently based on context, or default to a balanced configuration for general use (e.g., ₹50,000–₹60,000 budget, FHD display, 16 GB RAM, 512 GB SSD).

    **Keys and Tokenized Values:**

    1. Laptop Type:
        Gaming laptop → 1  
        Thin and light → 2  
        2 in 1 → 3  
        Notebook → 4  

    2. CPU Brand:
        Intel → 1  
        AMD → 2  
        Qualcomm → 3  
        Apple → 4  
        Mediatek → 5  

    3. RAM Type:
        LPDDR5 → 8  
        Unified Memory → 8  
        DDR5 → 7  
        LPDDR4X → 6  
        LPDDR4 → 5  
        DDR4 → 4  
        LPDDR3 → 2  
        DDR3 → 1  

    4. GPU (Boolean):
        0 → No Dedicated GPU  
        1 → Dedicated GPU  

    5. GPU Name (Tokenized): 
        nvidia geforce rtx 3070 ti --> 1
        nvidia geforce rtx 3050 --> 2
        intel hd --> 3
        iris xe --> 4
        amd radeon --> 5
        nvidia geforce gtx 1650 --> 6
        intel uhd --> 7
        intel iris xe --> 8
        nvidia geforce rtx 3070 --> 9
        nvidia geforce rtx 3050 ti --> 10
        amd radeon vega 8 --> 11
        nvidia geforce rtx 3060 --> 12
        nvidia geforce mx 450 --> 13
        intel uhd 600 --> 14
        amd radeon rx vega 10 --> 15
        qualcomm adreno 618 gpu --> 16
        amd radeon vega --> 17
        amd radeon rx 6600m --> 18
        qualcomm adreno --> 19
        nvidia geforce rtx 3050ti --> 20
        nvidia geforce rtx 3080 ti --> 21
        intel iris plus --> 22
        nvidia geforce gtx --> 23
        intel uhd 605 --> 24
        nvidia geforce rtx --> 25
        intel iris --> 26
        nvidia geforce mx 350 --> 27
        nvidia geforce mx 330 --> 28
        amd radeon rx 6700m --> 29
        nvidia geforce --> 30
        nvidia geforce gtx 1650 ti --> 31
        m1 --> 32
        amd radeon vega --> 33
        intel hd 520 --> 34
        amd radeon rx6600m --> 35
        amd radeon rx 6800m --> 36
        amd radeon 5500u --> 37
        amd radeon 5500m --> 38
        nvidia geforce mx 130 --> 39
        amd radeon r4 --> 40
        mediatek --> 41
        intel hd 500 --> 42
        amd radeon r5 --> 43
        amd radeon r4 (stoney ridge) --> 44
        nvidia geforce gtx 1650 max q --> 45
        nvidia geforce mx 250 --> 46
        nvidia geforce gtx 1660 ti --> 47
        nvidia geforce rtx 2060 --> 48
        intel uhd 620 --> 49
        intel hd 5500 --> 50
        nvidia geforce rtx 2080 super max-q --> 51
        amd radeon vega 6 --> 52
        intel iris xe max --> 53
        nvidia geforce gtx 1650 ti max-q --> 54
        amd radeon 520 --> 55
        nvidia geforce rtx 2070 max-q --> 56
        nvidia geforce gtx mx 330 --> 57
        nvidia geforce mx 230 --> 58
        intel hd 620 --> 59
        nvidia quadro p520 --> 60
        nvidia quadro t2000 --> 61
        nvidia geforce mx 110 --> 62
             
    6. Storage Type:
        0 → No SSD  
        1 → SSD  

    7. Operating System: 
        Windows → 1  
        Chrome OS → 2  
        DOS → 3  
        Mac → 4  
        Ubuntu → 5  

    8. Laptop Company:
        ASUS → 1  
        HP → 2  
        Lenovo → 3  
        Dell → 4  
        MSI → 5  
        Realme → 6  
        (Full list of laptop companies tokenized as above...)

    11. Battery Life (hours): Include as an float value.

    Example output:
    {{
        "Type": 1,
        "CPU Brand": 1,
        "RAM Type": 7,
        "GPU": 1,
        "GPU Name": 2,
        "Screen Resolution": 2,
        "Refresh Rate": 4,
        "Price Range": {{"min": 70000, "max": 100000}},
        "RAM": 16,
        "Storage": 512,
        "Storage Type": 1,
        "Operating System": 1,
        "Laptop Company": 4,
        "Battery Life": 6.0
    }}

    - Ensure outputs are strictly formatted as JSON. Do not provide explanations.
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
        ('human', "Using recommendations: {recommendations}, answer the user's query: {question}")
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

# Create chains with chat memory
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

# Utility functions
def get_recommendations(parsed_input):
    try:
        recommendations = find_best_recommendation(parsed_input)
        if "error" in recommendations:
            raise ValueError(recommendations["error"])
        return recommendations
        
    except Exception as e:
        return {"error": f"Recommendation error: {str(e)}"}

# Session-based state to track parsed input and recommendations
session_state = {}

def generate_response(user_message, session_id='default'):
    """
    Handle user message, maintain chat history, and generate response.
    """
    try:
        # Initialize session state if not already done
        if session_id not in session_state:
            session_state[session_id] = {
                "parsed_data": None,
                "recommendations": None,
            }

        # Retrieve session-specific data
        session = session_state[session_id]

        # If not parsed, parse user input
        if session["parsed_data"] is None:
            parsed_input = parse_chain_with_history.invoke(
                {'question': user_message, 'knowledge_base': knowledge_base},
                config={'configurable': {'session_id': session_id}}
            )
            parsed_input = strip_backticks(parsed_input)
            
            if "error" in parsed_input:
                return {"error": parsed_input["error"]}
            
            print("DEBUG: Parsed Input: ", parsed_input)

            # Validate and store parsed data
            try:
                session["parsed_data"] = json.loads(parsed_input)
            except json.JSONDecodeError:
                return "Sorry, I couldn't understand that. Could you provide more details about your requirements?"

            # Generate recommendations based on parsed input
            session["recommendations"] = get_recommendations(session["parsed_data"])
            if "error" in session["recommendations"]:
                return {"error": session["recommendations"]["error"]}
            print("DEBUG: Recommendations: ", session["recommendations"])

        # Use stored recommendations to generate the response
        response = response_chain_with_history.invoke(
            {
                'question': user_message,
                'recommendations': session["recommendations"]
            },
            config={'configurable': {'session_id': session_id}}
        )
        print("DEBUG: Response: ", response)
        return response.strip()

    except Exception as e:
        print("Error in generating response:", str(e))
        return "An error occurred while generating the response."

