import pandas as pd
import streamlit as st
from datetime import datetime
import PyPDF2
from google import genai
import json
from typing import List

# Set to display all columns
pd.set_option('display.max_columns', None)

# Set max rows to display (None means unlimited)
pd.set_option('display.max_rows', None)

# You can add your category_guide here if you re-introduce it
category_guide = """
- Living Expenses:
    - Rent & Mortgage Payments
    - Utilities (Electricity, Water, Gas)
    - Internet & Cable TV
    - Groceries - Supermarket Purchases
    - Dining Out - Restaurants & Cafes
    - Transportation - Fuel, Public Transport, Ride-sharing
    - Household Supplies
    - Home Maintenance & Repairs
- Personal & Lifestyle:
    - Clothing & Accessories
    - Personal Care (Haircuts, Cosmetics)
    - Entertainment (Movies, Concerts, Hobbies)
    - Fitness & Wellness (Gym Memberships, Sports)
    - Education & Books
    - Travel & Vacations
    - Gifts & Donations
- Financial:
    - Loan Payments (Student, Personal)
    - Credit Card Payments (Payments made to the card, not purchases)
    - Savings & Investments
    - Insurance (Health, Auto, Home)
    - Bank Fees
- Healthcare:
    - Doctor Visits & Medical Services
    - Pharmacy & Prescriptions
    - Dental Care
    - Vision Care
- Subscriptions:
    - Streaming Services (Netflix, Spotify)
    - Software Subscriptions
    - Magazine/Newspaper Subscriptions
    - Membership Fees (e.g., Amazon Prime, gym membership, but if the gym membership is under Fitness & Wellness, then put it there)
"""

@st.cache_data
def extract_text_and_tables_from_uploaded_pdfs(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    """
    Extracts all text and table data (as text) from Streamlit uploaded PDF files
    using PyPDF2.

    Args:
        uploaded_files (List[st.runtime.uploaded_file_manager.UploadedFile]):
            A list of Streamlit UploadedFile objects, each representing an uploaded PDF file.

    Returns:
        List[str]: A list of strings, where each string contains the extracted
                    text and table data from a single PDF file.
    """
    extracted_data = []

    if not uploaded_files:
        st.warning("No PDF files uploaded for extraction.")
        return []

    st.info(f"Found {len(uploaded_files)} PDF file(s) to process.")

    for uploaded_file in uploaded_files:
        text_from_pdf = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text_from_pdf += page.extract_text() + "\n"
            extracted_data.append(text_from_pdf)
            st.success(f"Extracted data from: **{uploaded_file.name}**")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    return extracted_data

@st.cache_data
def get_gemini_response_from_pdf_data(pdf_texts: list[str]) -> str:
    """
    Feeds the extracted PDF text data to the Gemini API and returns the response.

    Args:
        pdf_texts (list[str]): A list of strings, each containing text extracted from a PDF.

    Returns:
        str: The text response from the Gemini API, expected to be in JSON format.
    """
    if not pdf_texts:
        return "No PDF text provided to generate a response."

    # Using st.secrets to securely access API key
    client = genai.Client(api_key="AIzaSyCUUhK6bngglPni-WOPCmTINAetFiisbnk") # Ensure you have this set up in Streamlit secrets

    # Define prompt
    prompt = f"""
    You are an expert at extracting financial transaction data from bank statements.
    Below is text extracted from one or more bank statement PDFs.
    Your task is to extract all transaction details and present them in a structured JSON array.
    Each element in the array should represent a single transaction and contain the following fields.
    If a piece of information is not explicitly available in the provided text, use an empty string for that field.

    Here are the required columns and their descriptions:
    1.  customer_id: A unique ID to identify the bank customer.
    2.  f_name: Customer first name.
    3.  l_name: Customer last name.
    4.  address: Customer address.
    5.  transaction_date: The date the transaction occurred (e.g., 'Jan 01').
    6.  posting_date: The date the transaction was posted (e.g., 'Jan 02').
    7.  activity_description: A detailed description of the transaction (e.g., 'PURCHASE AT STARBUCKS').
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

    Here is the extracted text from the PDF(s):
    """ + "\n\n".join(pdf_texts)

    contents = [prompt]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Using a suitable model
            contents=contents,
        )

        if response and response.text:
            return response.text
        else:
            return "No response received from the model or empty response."

    except Exception as e:
        return f"Error feeding data to Gemini API: {str(e)}"

def convert_gemini_response_to_dataframe(response_text: str) -> pd.DataFrame:
    """
    Converts the text response from the Gemini API (expected to be JSON)
    into a pandas DataFrame.

    Args:
        response_text (str): The JSON string returned from the Gemini API.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the parsed transaction data.
    """
    if not response_text or "No response received" in response_text or "Error feeding data" in response_text:
        st.warning("No valid response from the AI model to convert to DataFrame.")
        return pd.DataFrame()

    try:
        # Remove the ```json and trailing ```
        json_str = response_text.strip('```json\n').strip('```').strip()

        # Parse the JSON string into a Python dictionary
        data = json.loads(json_str)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            st.warning(f"Unexpected JSON structure received from AI model: {type(data)}. Returning empty DataFrame.")
            return pd.DataFrame()

        st.success("Successfully converted AI response to DataFrame.")

        # --- Post-processing for DataFrame ---
        def parse_date_with_year(date_str, year=datetime.now().year): # Default to current year
            try:
                # Handle cases where year might be included or not
                if len(date_str.split()) == 3: # e.g., 'Jan 01 2024'
                    return datetime.strptime(date_str, "%b %d %Y")
                else: # e.g., 'Jan 01'
                    return datetime.strptime(f"{date_str} {year}", "%b %d %Y")
            except ValueError:
                return pd.NaT

        if 'transaction_date' in df.columns:
            df['transaction_date'] = df['transaction_date'].apply(parse_date_with_year)
        if 'posting_date' in df.columns:
            df['posting_date'] = df['posting_date'].apply(parse_date_with_year)

        # Extract month and day for analysis
        if not df.empty and 'transaction_date' in df.columns:
            df['month'] = df['transaction_date'].dt.month
            df['day'] = df['transaction_date'].dt.day
            df['month_name'] = df['transaction_date'].dt.strftime('%B')
            df['day_of_week'] = df['transaction_date'].dt.day_name()
            df['year'] = df['transaction_date'].dt.year

        if 'amount_spent' in df.columns:
            df['amount_spent'] = pd.to_numeric(df['amount_spent'], errors='coerce')
        else:
            st.warning("Warning: 'amount_spent' column not found in DataFrame.")

        return df

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from AI response: {e}")
        st.code(f"Attempted to decode (first 500 chars): \n{response_text[:500]}...", language="json")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during DataFrame conversion: {e}")
        return pd.DataFrame()

def render_metric_card(column, title, value, delta=None):
    with column:
        card_html = f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {"<div class='" + ("metric-delta positive" if "Credit" in str(delta) else "metric-delta") + "'>" + delta + "</div>" if delta else ""}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)