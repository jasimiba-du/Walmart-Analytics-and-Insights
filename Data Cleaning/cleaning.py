# Import required libraries
import pandas as pd
import numpy as np
import sqlalchemy as sal

# Read the data
df = pd.read_csv("Walmart.csv")
# Drop duplicates
df = df.drop_duplicates()

# Clean price data - remove $ and convert to float
df["unit_price"] = df["unit_price"].str.replace("$", "").astype(float)

# Calculate total price
df["total_price"] = df["unit_price"] * df["quantity"]

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y")

# Convert time string to proper time format
df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.time

# Convert column names to lowercase
df.columns = df.columns.str.lower()
# Drop rows with null values
df = df.dropna()
print("Number of rows after dropping nulls:", len(df))

# PostgreSQL connection
username = "jasim"
database = "wallmart_sales"
password = "ab123693"
host = "localhost"
port = "5432"

# Create SQLAlchemy engine
engine = sal.create_engine(
    f"postgresql+pg8000://{username}:{password}@{host}:{port}/{database}"
)

# Define data types for PostgreSQL
dtype_mapping = {
    "invoice_id": sal.Integer,
    "branch": sal.String,
    "city": sal.String,
    "category": sal.String,
    "unit_price": sal.Float,
    "quantity": sal.Float,
    "date": sal.Date,
    "time": sal.Time,
    "payment_method": sal.String,
    "rating": sal.Float,
    "profit_margin": sal.Float,
    "total_price": sal.Float,
}

# Upload to PostgreSQL
try:
    with engine.connect() as conn:
        df.to_sql(
            "sales_data",
            con=conn,
            index=False,
            if_exists="replace",
            dtype=dtype_mapping,
        )
        print("Data loaded successfully with proper date and time formats")
except Exception as e:
    print(f"Error occurred: {str(e)}")
