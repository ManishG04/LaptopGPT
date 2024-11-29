import pandas as pd
from chatbot import generate_response

# Load preprocessed data
laptop_data = pd.read_csv("data/numeric.csv")

def find_best_recommendation(user_input):
    """
    Find a single laptop matching user specifications using basic conditionals.
    """
    filtered_data = laptop_data

    # Apply filters based on user input
    if "Type" in user_input:
        filtered_data = filtered_data[filtered_data["Type"] == user_input["Type"]]

    if "Screen Size" in user_input:
        filtered_data = filtered_data[
            filtered_data["Screen Size (in inch)"].between(
                user_input["Screen Size"] - 1, user_input["Screen Size"] + 1
            )
        ]

    if "Price Range" in user_input:
        filtered_data = filtered_data[
            (filtered_data["Price (in Indian Rupees)"] >= user_input["Price Range"]["min"]) &
            (filtered_data["Price (in Indian Rupees)"] <= user_input["Price Range"]["max"])
        ]

    if "RAM" in user_input:
        filtered_data = filtered_data[filtered_data["RAM (in GB)"] >= user_input["RAM"]]

    # if "GPU" in user_input:
    #     filtered_data = filtered_data[filtered_data["GPU"] == user_input["GPU"]]

    # Return the first match or indicate no match found
    if not filtered_data.empty:
        return filtered_data.iloc[0].to_dict()
    else:
        return {"error": "No laptops found matching your criteria."}
