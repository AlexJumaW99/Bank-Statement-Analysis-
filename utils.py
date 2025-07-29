import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import PyPDF2
from google import genai
import json
from typing import List
import hashlib

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
    Feeds extracted PDF text to the Gemini API.
    The prompt is now simplified to only extract raw data, not perform calculations.
    """
    if not pdf_texts:
        return "[]"

    all_transactions = []
    try:
        client = genai.Client(api_key=st.secrets["gemini"]["api_key"])
    except Exception as e:
        st.error(f"Could not initialize Gemini client. Ensure your API key is in secrets.toml: [gemini] api_key='...'")
        return "[]"
    
    # caveat = "Skip rows for liquor, cannabis (e.g., Wowkpow, Toad In The Hole, Kings Head Pub, Canna Cabana), or cash advances. These should not be included in the final output."
    caveat = " "
    prompt = f"""
    You are an expert at extracting financial transaction data from bank statements.
    From the text below, extract all transaction details into a structured JSON array.
    Each object should represent one transaction.
    If information is not available, use an empty string or null.
    {caveat}

    Required columns:
    1.  customer_id: Unique customer ID (e.g., 'alex_juma').
    2.  f_name: Customer first name.
    3.  l_name: Customer last name.
    4.  address: Customer address (e.g 2525 PEMBINA HWY 603 WINNIPEG MB R3T 6H3).
    5.  transaction_date: Date of the transaction (MM-DD-YYYY format).
    6.  posting_date: Date the transaction was posted (MM-DD-YYYY format).
    7.  activity_description: Merchant name, simplified and in ALL CAPS (e.g., 'UBER').
    8.  category: Broad category for the transaction.
    9.  sub_category: Specific sub-category. Use this guide for categories: {category_guide}.
    10. amount_spent: Transaction amount. Positive for expenses, negative for payments/credits.
    11. credit_limit: The total credit limit on the statement.
    12. available_credit: The available credit listed at the START of the statement (extract only once).
    13. is_subscription: Boolean (true/false) for recurring subscriptions.

    Output must be a single, valid JSON array of objects.
    """
    
    for text in pdf_texts:
        try:
            full_prompt = prompt + "\n\nHere is the extracted text:\n" + text
            response = client.models.generate_content(model="models/gemini-1.5-flash", contents=[full_prompt])
            if response and response.text:
                json_str = response.text.strip().lstrip('```json').rstrip('```')
                transactions = json.loads(json_str)
                if isinstance(transactions, list):
                    all_transactions.extend(transactions)
        except Exception as e:
            st.warning(f"Could not process a document with AI. It might be a formatting issue. Error: {e}")
            continue
    
    st.json(all_transactions, expanded=True)
    return json.dumps(all_transactions) if all_transactions else "[]"


def convert_gemini_response_to_dataframe(response_text: str) -> pd.DataFrame:
    """
    Converts the JSON string from Gemini into a fully preprocessed DataFrame.
    This now includes hashing, cleaning, and calculating the running credit balance.
    """
    if not response_text: return pd.DataFrame()
    try:
        data = json.loads(response_text)
        if not isinstance(data, list) or not data: return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df = apply_data_types(df)
        df = create_transaction_hash(df)
        df = calculate_running_credit(df) # Calculate credit after cleaning and sorting
        return df
    
    except json.JSONDecodeError:
        st.error("Failed to decode the AI's response. The format was not valid JSON.")
        return pd.DataFrame()

def apply_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single source of truth for data cleaning and type conversion.
    """
    if df.empty: return df

    df.columns = df.columns.str.strip()

    for col in ['transaction_date', 'posting_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in ['amount_spent', 'credit_limit', 'available_credit']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'is_subscription' in df.columns:
        df['is_subscription'] = df['is_subscription'].apply(
            lambda x: str(x).lower() in ['true', '1', 't', 'y', 'yes'] if pd.notna(x) else False
        ).astype(bool)

    if 'transaction_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
        valid_dates = df['transaction_date'].dropna()
        df.loc[valid_dates.index, 'year'] = valid_dates.dt.year.astype('Int64')
        df.loc[valid_dates.index, 'month'] = valid_dates.dt.month.astype('Int64')
        df.loc[valid_dates.index, 'day'] = valid_dates.dt.day.astype('Int64')
        df.loc[valid_dates.index, 'month_name'] = valid_dates.dt.strftime('%B')
        df.loc[valid_dates.index, 'day_of_week'] = valid_dates.dt.day_name()

    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
    
    return df

def create_transaction_hash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a unique MD5 hash for each transaction row to prevent duplicates.
    """
    if df.empty:
        df['TransactionHash'] = pd.Series(dtype='str')
        return df

    def generate_hash(row):
        # Standardize data for consistent hashing
        date_str = str(row['transaction_date'].date()) if pd.notna(row['transaction_date']) else ''
        desc_str = str(row['activity_description']).lower().strip()
        amount_str = f"{row['amount_spent']:.2f}" if pd.notna(row['amount_spent']) else ''
        
        # Combine into a single string
        hash_string = f"{date_str}-{desc_str}-{amount_str}".encode('utf-8')
        return hashlib.md5(hash_string).hexdigest()

    df['TransactionHash'] = df.apply(generate_hash, axis=1)
    return df

def calculate_running_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accurately calculates the available_credit column after each transaction.
    This function assumes the DataFrame is for a single statement period.
    """
    if df.empty or 'amount_spent' not in df.columns or 'available_credit' not in df.columns:
        return df

    # Sort by transaction date to ensure correct calculation order
    df = df.sort_values(by='transaction_date', ascending=True).reset_index(drop=True)
    
    # Find the first valid 'available_credit' value, which should be the starting credit
    first_valid_idx = df['available_credit'].first_valid_index()
    
    if first_valid_idx is None:
        # If no starting credit is found, we cannot calculate the running balance.
        return df
        
    start_credit = df.loc[first_valid_idx, 'available_credit']
    
    # Calculate the cumulative change in balance. A "spent" amount (positive) decreases credit.
    # We use .iloc[first_valid_idx:] to start the calculation from the first transaction.
    cumulative_spend = df.loc[first_valid_idx:, 'amount_spent'].cumsum()
    
    # The running balance is the starting credit minus the cumulative spend up to that point.
    df.loc[first_valid_idx:, 'available_credit'] = start_credit - cumulative_spend
    
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
            model="models/gemini-1.5-flash",
            contents=[prompt],
        )
        return response.text if response and response.text else "No recommendations received from the model."
    except Exception as e:
        return f"Error communicating with the Gemini API: {str(e)}"



