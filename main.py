import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime



# Import functions from your utils.py file
from utils import (
    extract_text_and_tables_from_uploaded_pdfs,
    get_gemini_response_from_pdf_data,
    convert_gemini_response_to_dataframe,
    render_metric_card,
    category_guide, # You might want to keep this in utils if it's a global constant for the AI prompt
    save_pandas_df_as_json,
    get_gemini_recommendations_based_on_transactions
)

# st.logout()

def main():
    user = st.user
    # st.json(st.user)
    # user.logged_in = False
    # print(dict(user))

    if (not user.is_logged_in) or (dict(user) == {}):
        # Streamlit app title and logo
        _, col, _ = st.columns([1, 2, 1])
        col.image("./media/logo3.png", width=500)
        st.subheader("What are you wasting money on? Unsure? Let's find out!")

        if st.button("Login", use_container_width=True):
            st.login("google")
            # st.session_state.authenticated = True
            # st.success("You are now logged in!")

        # st.json(st.user)
        # print(st.user)

    elif user.is_logged_in:
        # st.json(user)

        st.set_page_config(layout="wide", page_title="Credit Card Dashboard")

        st.title("Credit Card Transaction Dashboard")
        st.subheader(f"Hello, {user.name}! Let's analyze your credit card transactions.")
        st.image(user.picture)

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
            list_of_pdf_files = uploaded_files

        # Initialize df as an empty DataFrame to prevent NameError if no files are uploaded
        df = pd.DataFrame()

        if list_of_pdf_files:
            with st.spinner("Extracting text from PDFs and analyzing with AI... This might take a moment."):
                extracted_text_list = extract_text_and_tables_from_uploaded_pdfs(list_of_pdf_files)
                if extracted_text_list:
                    response = get_gemini_response_from_pdf_data(extracted_text_list)
                    df = convert_gemini_response_to_dataframe(response)
                    save_pandas_df_as_json(df, "transaction_data.json")
                    # st.success("Data extraction and analysis complete! The dashboard is now populated with your credit card transactions.")
                else:
                    st.info("No text extracted from the uploaded PDFs. Please ensure they are valid and contain readable text.")
        else:
            # st.info("Please upload your PDF statement(s) to get started.")
            pass

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

            # Use the imported render_metric_card function
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

            # ------------------ AI RECOMMENDATIONS SECTION ------------------
            st.header("🤖 AI Recommendations")
            if st.button("Predict"):
                st.write(get_gemini_recommendations_based_on_transactions(df.to_json(orient='records', date_format='iso')))


        else:
            st.info("Please upload a bank statement PDF to populate the dashboard. Once uploaded, the extracted data and analysis will appear here.")

        if st.button("Logout", use_container_width=True):
            st.logout()

if __name__ == "__main__":
    main()
