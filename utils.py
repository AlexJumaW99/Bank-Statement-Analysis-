import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import PyPDF2
from google import genai
import json
from typing import List

# This guide remains the same.
category_guide = """
- Living Expenses: Rent, Utilities, Groceries, Dining Out, Transportation
- Personal & Lifestyle: Clothing, Personal Care, Entertainment, Fitness, Travel
- Financial: Loan Payments, Credit Card Payments, Savings, Insurance, Bank Fees
- Healthcare: Doctor Visits, Pharmacy, Dental, Vision
- Subscriptions: Streaming, Software, Memberships
- Amazon: All purchases made on Amazon, including physical goods and digital content, sub-category should also be Amazon.
- Other: Any transaction that does not fit into the above categories, sub-category should also be Other.
"""

# This custom CSS remains the same.
custom_css_markdown = """
        <style>
        .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); height: 150px; display: flex; flex-direction: column; justify-content: center; text-align: center;}
        .metric-card:hover {box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);}
        .metric-title { font-size: 1.1em; font-weight: bold; margin-bottom: 5px;}
        .metric-value { font-size: 2.2em; color: #007bff; font-weight: bolder;}
        .metric-delta {font-size: 0.9em; font-weight: bold;}
        .metric-delta.positive {color: #28a745;}
        .metric-delta.inverse {color: #dc3545;}
        </style>
        """

@st.cache_data
def extract_text_and_tables_from_uploaded_pdfs(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    extracted_data = []
    if not uploaded_files: return []
    for file in uploaded_files:
        try:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages)
            extracted_data.append(text)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
    return extracted_data


@st.cache_data
def get_gemini_response_from_pdf_data(pdf_texts: List[str]) -> str:
    """
    Feeds extracted PDF text to the Gemini API and aggregates responses.
    This function remains largely the same but ensures a robust prompt.
    """
    if not pdf_texts:
        return "[]"

    all_transactions = []
    # Use st.secrets for the API key for security
    try:
        client = genai.Client(api_key=st.secrets["gemini"]["api_key"])
    except Exception as e:
        st.error(f"Could not initialize Gemini client. Ensure your API key is in secrets.toml: [gemini] api_key='...'")
        return "[]"

    prompt = f"""
    You are an expert at extracting financial transaction data from bank statements.
    Below is text extracted from one or more bank statement PDFs.
    Your task is to extract all transaction details and present them in a structured JSON array.
    Each element in the array should represent a single transaction and contain the following fields.
    If a piece of information is not explicitly available in the provided text, use an empty string for that field.
    Please skip over any rows with liquor and cannabis purchases, as these are not relevant to the analysis. The merchants Wowkpow, Toad In The Hole, Kings Head Pub and Canna Cabana for example are cannabis stores and should be skipped. 
    Also skip rows with cash advances, as these are not relevant to the analysis. 

    Here are the required columns and their descriptions:
    1.  customer_id: A unique ID to identify the bank customer, it should consist of the first_name, followed by an underscore, then the last_name e.g alex_juma
    2.  f_name: Customer first name.
    3.  l_name: Customer last name.
    4.  address: Customer address.
    5.  transaction_date: The date the transaction occurred (e.g., 'Jan 01') should be written as 01-01-2024 (MM-DD-YYYY format).
    6.  posting_date: The date the transaction was posted (e.g., 'Jan 02') should be written as 01-01-2024 (MM-DD-YYYY format).
    7.  activity_description: Refers to the merchant that the user bought the good or service from (e.g., 'UBER* TRIP TORONTO ON'). Please simplify this merchant name to a simpler name if possible (e.g., 'UBER'). Merchant name should be in all caps always. 
    8.  category: A broad category for the transaction (e.g., 'Living Expenses', 'Personal & Lifestyle').
    9.  sub_category: A more specific sub-category for the transaction (e.g., 'Coffee Shops', 'Groceries - Supermarket Purchases').
    Please use the following as reference to come up with the categories and sub-categories: {category_guide}.
    10. amount_spent: The amount of money spent in the transaction. This should be a positive number for debits/expenses and a negative number for credits/returns.
    11. credit_limit: The total credit limit given by the bank, found on the statement.
    12. available_credit: The available credit after each transaction. The first row should have the initial value on the statement. For subsequent rows, update this value by adding the 'amount_spent' of the *current* transaction to the 'available_credit' of the *previous* row.
    13. is_subscription: A boolean (true/false) indicating if the transaction is for a recurring subscription service.

    Please ensure that 'amount_spent' is a numeric type (float or int) and 'is_subscription' is a boolean.
    The output must be a single JSON array of objects, with no additional text or formatting outside of the JSON.

    DO NOT STOP UNTIL THE FULL JSON ARRAY IS GENERATED.
    """
    
    for text in pdf_texts:
        try:
            full_prompt = prompt + "\n\nHere is the extracted text:\n" + text
            response = client.models.generate_content(model="gemini-2.5-flash", contents=[full_prompt])
            if response and response.text:
                json_str = response.text.strip().lstrip('```json').rstrip('```')
                transactions = json.loads(json_str)
                if isinstance(transactions, list):
                    all_transactions.extend(transactions)
        except Exception as e:
            st.warning(f"Could not process a document with AI. It might be a formatting issue. Error: {e}")
            continue
    
    return json.dumps(all_transactions) if all_transactions else "[]"


def convert_gemini_response_to_dataframe(response_text: str) -> pd.DataFrame:
    """
    Converts the JSON string from Gemini into a fully preprocessed DataFrame.
    This function now calls apply_data_types, making it the central point for creating
    a clean DataFrame ready for database insertion or visualization.
    """
    if not response_text: return pd.DataFrame()
    try:
        data = json.loads(response_text)
        if not isinstance(data, list) or not data: return pd.DataFrame()
        
        # This is the key step: convert raw data and apply all transformations
        return apply_data_types(pd.DataFrame(data))
    
    except json.JSONDecodeError:
        st.error("Failed to decode the AI's response. The format was not valid JSON.")
        return pd.DataFrame()
    

def apply_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is the single source of truth for data cleaning and feature engineering.
    It takes a raw DataFrame (from Gemini or the DB) and returns a clean,
    correctly-typed DataFrame ready for use.
    """
    if df.empty: return df

    # Convert date columns, coercing errors to NaT (Not a Time)
    for col in ['transaction_date', 'posting_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert numeric columns, coercing errors to NaN (Not a Number)
    for col in ['amount_spent', 'credit_limit', 'available_credit']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert boolean column
    if 'is_subscription' in df.columns:
        # Safely convert to boolean, handling various truthy/falsy values
        df['is_subscription'] = df['is_subscription'].apply(
            lambda x: str(x).lower() in ['true', '1', 't', 'y', 'yes'] if pd.notna(x) else False
        ).astype(bool)

    # --- Feature Engineering: Create new columns from the transaction date ---
    # This should only run if 'transaction_date' exists and is a datetime column
    if 'transaction_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
        # Fill missing dates temporarily to avoid errors, then drop them
        valid_dates = df['transaction_date'].dropna()
        df.loc[valid_dates.index, 'year'] = valid_dates.dt.year.astype('Int64') # Use nullable integer
        df.loc[valid_dates.index, 'month'] = valid_dates.dt.month.astype('Int64')
        df.loc[valid_dates.index, 'day'] = valid_dates.dt.day.astype('Int64')
        df.loc[valid_dates.index, 'month_name'] = valid_dates.dt.strftime('%B')
        df.loc[valid_dates.index, 'day_of_week'] = valid_dates.dt.day_name()

    # Clean up string columns by stripping whitespace
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
    
    # Strip whitespace from column names as well
    df.columns = df.columns.str.strip()
    
    return df


def render_metric_card(column, title, value, delta_value=None, delta_is_inverse=False):
    # This function remains the same.
    with column:
        delta_html = ""
        if delta_value:
            delta_class = "metric-delta inverse" if delta_is_inverse else "metric-delta positive"
            delta_html = f"<div class='{delta_class}'>{delta_value}</div>"
        
        card_html = f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    

def get_gemini_recommendations_based_on_transactions(transactions_json: str) -> str:
    # This function remains the same.
    try:
        client = genai.Client(api_key=st.secrets["gemini"]["api_key"])
    except Exception as e:
        st.error(f"Could not initialize Gemini client. Ensure your API key is in secrets.toml.")
        return "Could not generate recommendations."

    prompt = f"Based on these transactions: {transactions_json}, provide a detailed tabular analysis of spending habits. Explain where money can be saved and suggest specific, actionable steps to reduce unnecessary expenses. Format the response in clear, easy-to-understand Markdown."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
        )
        return response.text if response and response.text else "No recommendations received from the model."
    except Exception as e:
        return f"Error communicating with the Gemini API: {str(e)}"
