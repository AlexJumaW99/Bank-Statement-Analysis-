import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Import the revised and new functions
from db_operations import (
    connect_to_db, create_tables, upsert_google_user,
    get_user_transactions, bulk_insert_transactions
)
from utils import (
    extract_text_and_tables_from_uploaded_pdfs, get_gemini_response_from_pdf_data,
    convert_gemini_response_to_dataframe, render_metric_card, custom_css_markdown,
    apply_data_types, get_gemini_recommendations_based_on_transactions
)

def main():
    st.set_page_config(layout="wide", page_title="Credit Card Dashboard")

    # Initialize DB connection and create tables if they don't exist
    if 'db_conn' not in st.session_state:
        st.session_state.db_conn = connect_to_db()
        if st.session_state.db_conn:
            create_tables(st.session_state.db_conn)
        else:
            st.error("Fatal: Could not connect to the database. App cannot continue.")
            st.stop()

    user = st.user

    if user and user.is_logged_in:
        # --- Logged-In User Experience ---
        # Upsert user info and store their ID in the session state
        if 'user_id' not in st.session_state:
            st.session_state.user_id = upsert_google_user(
                st.session_state.db_conn, user.email, user.name, user.picture
            )
            if st.session_state.user_id is None:
                st.error("Fatal: Failed to retrieve or create a user profile in the database.")
                st.stop()

        # Load user's transactions from the database on first run
        if 'transactions_df' not in st.session_state:
            with st.spinner("Loading transaction history..."):
                # get_user_transactions now returns a DataFrame directly
                df_from_db = get_user_transactions(st.session_state.db_conn, st.session_state.user_id)
                if not df_from_db.empty:
                    # It's still crucial to run apply_data_types to ensure
                    # columns are in the correct format for visualization (e.g., datetime objects)
                    st.session_state.transactions_df = apply_data_types(df_from_db)
                    st.success("Transaction history loaded!")
                else:
                    # Initialize with an empty DataFrame if no history exists
                    st.session_state.transactions_df = pd.DataFrame()

        df = st.session_state.transactions_df

        # --- Sidebar ---
        with st.sidebar:
            st.subheader(f"Hello, {user.name}!")
            st.image(user.picture, width=100)
            if st.button("Logout", use_container_width=True):
                st.logout()

            st.header("Upload PDF Statement(s)")
            uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

            if uploaded_files and st.button("Process Uploaded Files", use_container_width=True):
                with st.spinner("Processing documents... This may take a moment."):
                    texts = extract_text_and_tables_from_uploaded_pdfs(uploaded_files)
                    if texts:
                        response = get_gemini_response_from_pdf_data(texts)
                        st.info("Gemini JSON Response:")
                        st.markdown(response)
                        # This function now returns a fully preprocessed DataFrame
                        new_df = convert_gemini_response_to_dataframe(response)
                        
                        if not new_df.empty:
                            # st.info("Extracted transactions via Gemini:")
                            # st.dataframe(new_df)    
                            # st.dataframe(new_df.dtypes)
                            # --- Deduplication Logic ---
                            def create_unique_id(d):
                                # Create a consistent, unique ID for each transaction to prevent duplicates
                                date_str = d['transaction_date'].dt.strftime('%Y-%m-%d').fillna('no_date')
                                desc_str = d['activity_description'].str.lower().str.strip().fillna('no_desc')
                                amount_str = d['amount_spent'].round(2).astype(str).fillna('no_amount')
                                return date_str + '-' + desc_str + '-' + amount_str
                            
                            new_df['unique_id'] = create_unique_id(new_df)
                            
                            existing_ids = set()
                            if not df.empty and 'transaction_date' in df.columns:
                                df['unique_id'] = create_unique_id(df)
                                existing_ids = set(df['unique_id'])
                            
                            df_to_insert = new_df[~new_df['unique_id'].isin(existing_ids)].drop(columns=['unique_id'])
                            
                            num_dupes = len(new_df) - len(df_to_insert)
                            if num_dupes > 0:
                                st.info(f"Skipped {num_dupes} duplicate transaction(s).")
                            
                            if not df_to_insert.empty:
                                # st.dataframe(df_to_insert)
                                # st.dataframe(df_to_insert.info())
                                # st.info("New transactions to be added:")
                                # st.dataframe(df_to_insert)
                                # st.dataframe(df_to_insert.dtypes)
                                # --- Perform Bulk Insert ---
                                # This is the new, efficient way to add data.
                                bulk_insert_transactions(st.session_state.db_conn, st.session_state.user_id, df_to_insert)
                                
                                # Update the session state and rerun to show new data
                                st.session_state.transactions_df = pd.concat([df.drop(columns=['unique_id'], errors='ignore'), df_to_insert], ignore_index=True)
                                st.success(f"Successfully added {len(df_to_insert)} new transactions!")
                                st.rerun()
                            elif num_dupes > 0:
                                st.warning("All transactions from the file(s) already exist in your history.")
                        else:
                            st.error("Could not extract any valid transactions from the PDF(s).")
                    else:
                        st.error("Failed to extract any text from the provided PDF(s).")
            
            if not df.empty:
                st.header("Filter by Time Period")
                all_years = sorted(df['year'].dropna().unique().astype(int))
                selected_years = st.multiselect("Select Year(s)", options=all_years, default=all_years)
                
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                available_months = sorted(df['month_name'].dropna().unique(), key=lambda m: month_order.index(m))
                selected_months = st.multiselect("Select Month(s)", options=available_months, default=available_months)
        
        # --- Main Dashboard Display ---
        st.title("💳 Credit Card Transaction Dashboard")
        st.markdown(custom_css_markdown, unsafe_allow_html=True)
        
        if not df.empty:
            # Filter data based on sidebar selections
            filtered_df = df[df['year'].isin(selected_years) & df['month_name'].isin(selected_months)].copy() if selected_years and selected_months else pd.DataFrame()

            if not filtered_df.empty:
                st.header("💾 Transaction Details")
                st.dataframe(filtered_df.drop(columns=['unique_id'], errors='ignore'))
                
                # --- The rest of your visualization code remains the same ---
                # It will now work reliably with the clean `filtered_df`
                expenses = filtered_df[filtered_df['amount_spent'] > 0]
                payments = filtered_df[filtered_df['amount_spent'] < 0].copy()
                payments['amount_spent'] = payments['amount_spent'].abs()

                st.header("📊 Overview")
                col1, col2, col3, col4 = st.columns(4)
                total_expenses = expenses['amount_spent'].sum()
                total_payments = payments['amount_spent'].sum()
                balance = total_expenses - total_payments
                avg_daily_spend = filtered_df.groupby('transaction_date')['amount_spent'].mean().mean()
                # st.write(avg_daily_spend)
                render_metric_card(col1, "Total Spending", f"${total_expenses:,.2f}")
                render_metric_card(col2, "Total Payments", f"${total_payments:,.2f}")
                render_metric_card(col3, "Current Balance", f"${balance:,.2f}", f"{'+' if balance >= 0 else ''}${balance:,.2f} {'(Owing)' if balance > 0 else '(Credit)'}", balance > 0)
                render_metric_card(col4, "Average Daily Spend", f"${avg_daily_spend:,.2f}", f"{'Net Debt Payer' if avg_daily_spend < 0 else 'Net Debt Borrower'}")


                st.header("💸 Spending Patterns")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top 10 Largest Transactions")
                    top10 = expenses.nlargest(10, 'amount_spent').reset_index()
                    # Create a unique label for each transaction to prevent grouping
                    top10['unique_label'] = top10['activity_description'] + " (" + top10['transaction_date'].dt.strftime('%b %d') + ")"
                    fig_top10 = px.bar(top10, x='amount_spent', y='unique_label', color='category', orientation='h', title='Top 10 Largest Transactions')
                    fig_top10.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_top10, use_container_width=True)

                with col2:
                    st.subheader("Daily Spending vs Payments")
                    # st.markdown("Compare your spending and payments over time.")
                    daily_spend = expenses.groupby('transaction_date')['amount_spent'].sum().reset_index()
                    daily_payments = payments.groupby('transaction_date')['amount_spent'].sum().abs().reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily_spend['transaction_date'], y=daily_spend['amount_spent'],
                                                     mode='lines', name='Daily Spending'))
                    fig.add_trace(go.Scatter(x=daily_payments['transaction_date'], y=daily_payments['amount_spent'],
                                                     mode='lines', name='Daily Payments'))
                    fig.update_layout(title='Daily Spending vs Payments', xaxis_title='Date', yaxis_title='Amount ($)')
                    st.plotly_chart(fig, use_container_width=True)


                st.header("📂 Category Breakdown")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Spending by Category")
                    category_totals = expenses.groupby('category')['amount_spent'].sum()
                    fig_cat = px.pie(values=category_totals.values, names=category_totals.index, title='Spending by Category')
                    st.plotly_chart(fig_cat, use_container_width=True)
                with col2:
                    st.subheader("Spending by Sub-Category")
                    sub_cat_totals = expenses.groupby('sub_category')['amount_spent'].sum().nlargest(10)
                    fig_sub_cat = px.pie(values=sub_cat_totals.values, names=sub_cat_totals.index, title='Top 10 Sub-Categories by Spending')
                    st.plotly_chart(fig_sub_cat, use_container_width=True)

                st.header("📅 Temporal Analysis")
                # Toggle for Monthly and Daily charts
                chart_type = st.radio("Select data to view:", ('Expenses', 'Payments'), horizontal=True, key='temporal_toggle')
                data_to_plot = expenses if chart_type == 'Expenses' else payments
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"{chart_type} by Month")
                    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                    monthly_totals = data_to_plot.groupby('month_name')['amount_spent'].sum().reindex(month_order).dropna()
                    fig_month = px.bar(monthly_totals, x=monthly_totals.index, y=monthly_totals.values, labels={'y': 'Amount ($)', 'x': 'Month'})
                    st.plotly_chart(fig_month, use_container_width=True)
                with col2:
                    st.subheader(f"{chart_type} by Day of Week")
                    day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                    dow_totals = data_to_plot.groupby('day_of_week')['amount_spent'].sum().reindex(day_order).dropna()
                    fig_dow = px.bar(dow_totals, x=dow_totals.index, y=dow_totals.values, labels={'y': 'Amount ($)', 'x': 'Day of Week'})
                    st.plotly_chart(fig_dow, use_container_width=True)

                st.header("🏪 Frequent Merchants")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top 10 Merchants by Transaction Count")
                    top_merchants = expenses['activity_description'].value_counts().head(10).reset_index()
                    top_merchants.columns = ['Merchant', 'Transaction Count']
                    st.dataframe(top_merchants)

                with col2:
                    st.subheader("Monthly Subcriptions Costs")
                    subscriptions = expenses[expenses['is_subscription'] == True] # Use the boolean column
                    # sorted_subscriptions = subscriptions.sort_values(by='amount_spent', ascending=False)
                    sorted_subscriptions = subscriptions.groupby('activity_description')['amount_spent'] \
                                  .sum() \
                                  .reset_index() \
                                  .sort_values(by='amount_spent', ascending=False)
                    if not sorted_subscriptions.empty:
                        st.dataframe(sorted_subscriptions[['activity_description', 'amount_spent']])
                    else:
                        st.info("No subscription costs for the selected filters.")



                st.header("🤖 AI Recommendations")
                if st.button("Generate Spending Analysis"):
                    with st.spinner("Analyzing your spending habits..."):
                        recs = get_gemini_recommendations_based_on_transactions(filtered_df.to_json(orient='records', date_format='iso'))
                        st.markdown(recs)
            else:
                st.warning("No data available for the selected filters. Please adjust your selection or upload a statement.")

        else:
            st.info("👋 Welcome! No transaction data found. Please upload a PDF statement using the sidebar to get started.")

    # --- Login Screen ---
    elif user:
        _, col, _ = st.columns([1, 2, 1])
        # Make sure you have a logo file at this path in your project directory
        # col.image("./media/logo3.png", width=500) 
        col.title("Financial Insights Dashboard")
        st.subheader("What are you wasting money on? Unsure? Let's find out!")
        if st.button("Login with Google", use_container_width=True):
            st.login("google")
    else:
        st.info("Authentication is not available. This can happen if the app is not running on a supported platform like Streamlit Community Cloud.")

if __name__ == "__main__":
    main()
