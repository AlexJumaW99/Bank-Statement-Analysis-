import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import PyPDF2
import plotly.express as px
import plotly.graph_objects as go

from typing import List
from io import BytesIO

from google import genai
from google.genai import types
import pathlib
from pathlib import Path
import os
import json

# Set to display all columns
pd.set_option('display.max_columns', None)

# Set max rows to display (None means unlimited)
pd.set_option('display.max_rows', None)

# It's recommended to load your API key securely, e.g., from Streamlit secrets
# genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

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
    client = genai.Client(api_key="AIzaSyCUUhK6bngglPni-WOPCmTINAetFiisbnk")

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
        def parse_date_with_year(date_str, year=2025):
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


def main():
    st.set_page_config(layout="wide", page_title="Credit Card Dashboard")
    st.title("Credit Card Transaction Dashboard")

    # ------------------ Custom CSS for Cards ------------------
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6; /* Light grey background */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
        height: 120px; /* Fixed height for uniformity */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .metric-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .metric-title {
        font-size: 1.1em;
        color: #333333;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .metric-value {
        font-size: 2.2em;
        color: #007bff; /* Blue for values */
        font-weight: bolder;
    }
    .metric-delta {
        font-size: 0.9em;
        color: #dc3545; /* Red for inverse delta (owing) */
        font-weight: bold;
    }
    .metric-delta.positive {
        color: #28a745; /* Green for positive deltas (if applicable) */
    }
    </style>
    """, unsafe_allow_html=True)

    # --- PDF Upload Section in Sidebar ---
    st.sidebar.header("Upload PDF Statement(s)")
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF file(s)",
        type="pdf",
        accept_multiple_files=True,
        help="Upload your credit card statement(s) in PDF format. You can upload multiple files at once."
    )

    list_of_pdf_files = []
    if uploaded_files:
        list_of_pdf_files = uploaded_files # Streamlit's file_uploader returns a list of UploadedFile objects directly

    # Initialize df as an empty DataFrame to prevent NameError if no files are uploaded
    df = pd.DataFrame()

    if list_of_pdf_files:
        with st.spinner("Extracting text from PDFs and analyzing with AI... This might take a moment."):
            extracted_text_list = extract_text_and_tables_from_uploaded_pdfs(list_of_pdf_files)
            if extracted_text_list: # Only proceed if text was successfully extracted
                response = get_gemini_response_from_pdf_data(extracted_text_list)
                df = convert_gemini_response_to_dataframe(response)
            else:
                st.info("No text extracted from the uploaded PDFs. Please ensure they are valid and contain readable text.")
    else:
        st.info("Please upload your PDF statement(s) to get started.")

    # Only display the dashboard if the DataFrame is not empty
    if not df.empty:
        # ------------------ TABLE SECTION ------------------
        st.header("💾 Raw Data")
        st.markdown("The data was successfully extracted from the uploaded PDF(s) and processed. Below is the raw data in tabular format.")
        st.dataframe(df, use_container_width=True)

        # --- FILTERS (New Section) ---
        st.sidebar.header("Filter by Time Period")

        # Year dropdown
        all_years = sorted(df['year'].unique())
        selected_years = st.sidebar.multiselect(
            "Select Year(s)",
            options=all_years,
            default=all_years,
            help="Select one or more years to filter the data."
        )

        # Month dropdown
        all_months = df['month_name'].unique()
        month_order = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        all_months_sorted = [month for month in month_order if month in all_months]

        selected_months = st.sidebar.multiselect(
            "Select Month(s)",
            options=all_months_sorted,
            default=all_months_sorted,
            help="Select one or more months to filter the data."
        )

        # Apply filters to the DataFrame
        filtered_df = df[
            df['year'].isin(selected_years) &
            df['month_name'].isin(selected_months)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Re-separate expenses and payments based on the filtered DataFrame
        filtered_expenses = filtered_df[filtered_df['amount_spent'] > 0]
        filtered_payments = filtered_df[filtered_df['amount_spent'] < 0]

        # ------------------ OVERVIEW SECTION ------------------
        st.header("📊 Overview")
        st.markdown("An at-a-glance summary of your overall credit card activity, including total spending, payments made, current balance, average daily spending, and top category of spending.")

        # Calculate metrics using filtered data
        total_expenses = filtered_expenses['amount_spent'].sum() if not filtered_expenses.empty else 0
        total_payments = abs(filtered_payments['amount_spent'].sum()) if not filtered_payments.empty else 0
        total_owing = total_expenses - total_payments
        avg_daily_spend = filtered_expenses['amount_spent'].mean() if not filtered_expenses.empty else 0
        
        if not filtered_expenses.empty:
            most_frequent_category = filtered_expenses['category'].value_counts().idxmax()
        else:
            most_frequent_category = "N/A"

        col1, col2, col3, col4, col5 = st.columns(5)

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

        render_metric_card(col1, "Total Spending", f"${total_expenses:,.2f}")
        render_metric_card(col2, "Total Payments", f"${total_payments:,.2f}")
        
        delta_class = "metric-delta"
        if total_owing > 0:
            delta_value = f"+${total_owing:,.2f} (Owing)"
            delta_class += " inverse"
        elif total_owing < 0:
            delta_value = f"-${abs(total_owing):,.2f} (Credit)"
            delta_class += " positive"
        else:
            delta_value = "$0.00"

        with col3:
            card_html_owing = f"""
            <div class="metric-card">
                <div class="metric-title">Current Balance</div>
                <div class="metric-value">${total_owing:,.2f}</div>
                <div class='{delta_class}'>{delta_value}</div>
            </div>
            """
            st.markdown(card_html_owing, unsafe_allow_html=True)

        render_metric_card(col4, "Avg Daily Spend", f"${avg_daily_spend:,.2f}")
        render_metric_card(col5, "Most Frequent Category", f"{most_frequent_category}")

        # ------------------ SPENDING PATTERNS SECTION ------------------
        st.header("💸 Spending Patterns")
        st.markdown("This section highlights trends in your daily and largest transactions.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 Largest Transactions")
            st.markdown("These are your biggest spending transactions, grouped by activity and category.")
            if not filtered_expenses.empty:
                top10 = filtered_expenses.nlargest(10, 'amount_spent')
                top10 = top10.sort_values(by='amount_spent', ascending=True)
                top10['unique_description'] = top10['activity_description'] + ' (ID:' + top10.index.astype(str) + ')'
                fig = px.bar(top10, x='amount_spent', y='unique_description', color='category', orientation='h',
                                title='Top 10 Largest Transactions')
                fig.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': top10['unique_description'].tolist()})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters to show top transactions.")

        with col2:
            st.subheader("Daily Spending vs Payments")
            st.markdown("Compare your spending and payments over time.")
            if not filtered_df.empty:
                daily_spend = filtered_expenses.groupby('transaction_date')['amount_spent'].sum().reset_index()
                daily_payments = filtered_payments.groupby('transaction_date')['amount_spent'].sum().abs().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_spend['transaction_date'], y=daily_spend['amount_spent'],
                                         mode='lines', name='Daily Spending'))
                fig.add_trace(go.Scatter(x=daily_payments['transaction_date'], y=daily_payments['amount_spent'],
                                         mode='lines', name='Daily Payments'))
                fig.update_layout(title='Daily Spending vs Payments', xaxis_title='Date', yaxis_title='Amount ($)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters to show daily trends.")

        # ------------------ CATEGORY BREAKDOWN SECTION ------------------
        st.header("📂 Category Breakdown")
        st.markdown("A look at your spending broken down by categories and sub-categories.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Spending by Category")
            st.markdown("Shows total spend distribution across main categories.")
            if not filtered_expenses.empty:
                category_totals = filtered_expenses.groupby('category')['amount_spent'].sum().reset_index()
                fig = px.pie(category_totals, names='category', values='amount_spent',
                                title='Spending by Category')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters to show category breakdown.")

        with col2:
            st.subheader("Top 10 Sub-Categories")
            st.markdown("The top sub-categories you spent the most on.")
            if not filtered_expenses.empty:
                subcat_totals = filtered_expenses.groupby('sub_category')['amount_spent'].sum().nlargest(10).reset_index()
                fig = px.pie(subcat_totals, names='sub_category', values='amount_spent',
                                title='Top 10 Sub-Category Spend Breakdown')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters to show top sub-categories.")

        # ------------------ TEMPORAL ANALYSIS SECTION ------------------
        st.header("📅 Temporal Analysis")
        st.markdown("Understand how your spending evolves across time and days.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Spending by Month")
            st.markdown("Bar chart comparing total spending for each month.")
            if not filtered_expenses.empty:
                monthly_totals = filtered_expenses.groupby('month_name')['amount_spent'].sum().reset_index()
                monthly_totals['month_name'] = pd.Categorical(
                    monthly_totals['month_name'], categories=month_order, ordered=True
                )
                monthly_totals = monthly_totals.sort_values(by='month_name')
                fig = px.bar(monthly_totals, x='month_name', y='amount_spent',
                                title='Monthly Spending', labels={'amount_spent': 'Amount ($)'})
                fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': monthly_totals['month_name'].tolist()})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters to show monthly spending.")

        with col2:
            st.subheader("Spending by Day of Week")
            st.markdown("Bar chart showing which days you tend to spend the most.")
            if not filtered_expenses.empty:
                dow_totals = filtered_expenses.groupby('day_of_week')['amount_spent'].sum().reset_index()
                day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                dow_totals['day_of_week'] = pd.Categorical(
                    dow_totals['day_of_week'], categories=day_order, ordered=True
                )
                dow_totals = dow_totals.sort_values(by='day_of_week')
                fig = px.bar(dow_totals, x='day_of_week', y='amount_spent',
                                title='Spending by Day of Week', labels={'amount_spent': 'Amount ($)'})
                fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': dow_totals['day_of_week'].tolist()})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for the selected filters to show daily spending.")

        # ------------------ CATEGORY-SPECIFIC ANALYSIS SECTION ------------------
        st.header("🩺 Health & Subscriptions")
        st.markdown("Dive into specific types of spending such as healthcare and subscriptions.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Health Expenses Over Time")
            st.markdown("Track how your health-related spending changes daily.")
            if not filtered_expenses.empty:
                health_expenses = filtered_expenses[filtered_expenses['category'] == 'Healthcare']
                if not health_expenses.empty:
                    health_daily = health_expenses.groupby('transaction_date')['amount_spent'].sum().reset_index()
                    fig = px.line(health_daily, x='transaction_date', y='amount_spent', title='Daily Health Expenses')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No health expenses for the selected filters.")
            else:
                st.info("No data available for the selected filters.")

        with col2:
            st.subheader("Subscription Costs")
            st.markdown("A table of all your subscription-based transactions.")
            if not filtered_expenses.empty:
                subscriptions = filtered_expenses[filtered_expenses['is_subscription'] == True] # Use the boolean column
                if not subscriptions.empty:
                    st.dataframe(subscriptions[['activity_description', 'amount_spent']])
                else:
                    st.info("No subscription costs for the selected filters.")
            else:
                st.info("No data available for the selected filters.")

        # ------------------ MERCHANT INSIGHTS SECTION ------------------
        st.header("🏪 Frequent Merchants")
        st.markdown("These are the merchants you transacted with the most.")

        if not filtered_expenses.empty:
            top_merchants = filtered_expenses['activity_description'].value_counts().head(10).reset_index()
            top_merchants.columns = ['Merchant', 'Transaction Count']
            st.dataframe(top_merchants)
        else:
            st.info("No data available for the selected filters.")
    else:
        st.info("Please upload a bank statement PDF to populate the dashboard. Once uploaded, the extracted data and analysis will appear here.")


if __name__ == "__main__":
    main()