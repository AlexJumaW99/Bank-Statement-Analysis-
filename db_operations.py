import pyodbc
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

def connect_to_db():
    """
    Establishes a connection to the SQL Server database using credentials from st.secrets.
    """
    try:
        db_config = st.secrets["database"]
        connection_string = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={db_config['server']};"
            f"DATABASE={db_config['database']};"
            f"UID={db_config['uid']};"
            f"PWD={db_config['pwd']};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=yes;"
        )
        conn = pyodbc.connect(connection_string, autocommit=False)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.warning("Please ensure your database credentials are correctly configured in `.streamlit/secrets.toml`.")
        return None

def create_tables(conn):
    """
    Creates the Users and Transactions tables in the database if they do not already exist.
    The schema is updated to include TransactionHash with a UNIQUE constraint for robust duplicate prevention.
    """
    try:
        cursor = conn.cursor()
        # Create Users table
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Users' AND xtype='U')
            CREATE TABLE Users (
                UserID INT PRIMARY KEY IDENTITY(1,1),
                GoogleEmail NVARCHAR(255) UNIQUE NOT NULL,
                DisplayName NVARCHAR(255),
                PictureURL NVARCHAR(MAX)
            )
        """)
        # Create Transactions table with a new TransactionHash column for duplicate detection
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Transactions' AND xtype='U')
            CREATE TABLE Transactions (
                TransactionID INT PRIMARY KEY IDENTITY(1,1),
                UserID INT NOT NULL,
                TransactionHash NVARCHAR(32) NOT NULL, -- MD5 hash is 32 chars
                customer_id NVARCHAR(255),
                f_name NVARCHAR(100),
                l_name NVARCHAR(100),
                address NVARCHAR(500),
                transaction_date DATETIME,
                posting_date DATETIME,
                activity_description NVARCHAR(MAX),
                category NVARCHAR(100),
                sub_category NVARCHAR(100),
                amount_spent DECIMAL(18, 2),
                credit_limit DECIMAL(18, 2),
                available_credit DECIMAL(18, 2),
                is_subscription BIT,
                month INT,
                day INT,
                month_name NVARCHAR(20),
                day_of_week NVARCHAR(20),
                year INT,
                FOREIGN KEY (UserID) REFERENCES Users(UserID),
                CONSTRAINT UQ_User_Transaction UNIQUE (UserID, TransactionHash) -- Enforce uniqueness per user
            )
        """)
        conn.commit()
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        st.error(f"Error creating tables: {sqlstate} - {ex}")
        conn.rollback()

def upsert_google_user(conn, email, display_name, picture_url):
    """
    Inserts or updates a user and returns their full profile as a dictionary.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT UserID FROM Users WHERE GoogleEmail = ?", (email,))
        result = cursor.fetchone()

        if result:
            cursor.execute("UPDATE Users SET DisplayName = ?, PictureURL = ? WHERE GoogleEmail = ?",
                           (display_name, picture_url, email))
        else:
            cursor.execute("INSERT INTO Users (GoogleEmail, DisplayName, PictureURL) VALUES (?, ?, ?)",
                           (email, display_name, picture_url))
        
        cursor.execute("SELECT UserID, DisplayName, PictureURL FROM Users WHERE GoogleEmail = ?", (email,))
        user_profile = cursor.fetchone()
        conn.commit()

        if user_profile:
            return {
                'UserID': user_profile.UserID,
                'DisplayName': user_profile.DisplayName,
                'PictureURL': user_profile.PictureURL
            }
        return None
        
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        st.error(f"Error upserting Google user: {sqlstate} - {ex}")
        conn.rollback()
        return None

def bulk_insert_transactions(conn, user_id, df):
    """
    Bulk inserts preprocessed transaction data from a DataFrame.
    It now includes the TransactionHash for each record.
    """
    if df.empty:
        st.info("No new transactions to insert.")
        return

    try:
        cursor = conn.cursor()

        # Define the columns in the order they appear in the Transactions table
        # This now includes TransactionHash.
        cols = [
            'TransactionHash', 'customer_id', 'f_name', 'l_name', 'address', 'transaction_date', 'posting_date',
            'activity_description', 'category', 'sub_category', 'amount_spent', 'credit_limit',
            'available_credit', 'is_subscription', 'month', 'day', 'month_name', 'day_of_week', 'year'
        ]
        
        # Prepare the DataFrame for insertion
        df_insert = df[cols].copy()
        df_insert.insert(0, 'UserID', user_id) # Add UserID to the beginning

        # Replace any remaining pandas-specific NA/NaN values with None for SQL NULL
        df_insert = df_insert.replace({pd.NaT: None, np.nan: None})

        # We will convert all numeric columns to strings before insertion
        # Not to worry though, the database constraints will handle type validation for these columns before insertion.
        for col in ['UserID', 'year', 'month', 'day', 'is_subscription', 'transaction_date', 'posting_date']:
            if col in df_insert.columns:
                df_insert[col] = df_insert[col].astype(object)

        # Convert DataFrame to a list of tuples, which is what executemany expects.
        # This handles numpy types correctly.
        data_tuples = [tuple(x) for x in df_insert.to_records(index=False)]

        # Construct the SQL INSERT statement dynamically
        sql = f"""
            INSERT INTO Transactions (
                UserID, {', '.join(cols)}
            ) VALUES (
                ?{', ?' * len(cols)}
            )
        """

        # Execute the bulk insert
        cursor.fast_executemany = True
        cursor.executemany(sql, data_tuples)
        
        conn.commit()

    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        st.error(f"Database error during bulk insert: {sqlstate} - {ex}")
        st.info("This might happen if you are trying to re-insert transactions that already exist.")
        st.error("The data that failed to insert:")
        st.dataframe(df_insert)
        st.dataframe(df_insert.dtypes)
        conn.rollback()


def get_user_transactions(conn, user_id):
    """
    Fetches all transaction data for a given UserID directly into a pandas DataFrame.
    """
    sql = "SELECT * FROM Transactions WHERE UserID = ?"
    try:
        df = pd.read_sql(sql, conn, params=[user_id])
        return df
    except Exception as ex:
        st.error(f"Error fetching transactions: {ex}")
        return pd.DataFrame()
