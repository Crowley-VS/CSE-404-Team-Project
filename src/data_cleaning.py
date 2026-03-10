"""
file: data_cleaning.py
output:
  - cleaned_retail.csv: cleaned transaction-level data
"""

import pandas as pd

def load_data(filepath: str):
    ext = filepath.rsplit(".", 1)[-1].lower()
    if ext in ("xlsx", "xls"):
        data = pd.read_excel(filepath, dtype={"InvoiceNo": str, "StockCode": str})
    else:
        data = pd.read_csv(filepath, dtype={"InvoiceNo": str, "StockCode": str})
    return data


def clean_data(data: pd.DataFrame):

    # duplicate rows
    data = data.drop_duplicates()

    # missing CustomerID 
    data = data.dropna(subset=["CustomerID"])

    # cancelled 
    data = data[~data["InvoiceNo"].str.startswith("C")]

    # remove negative/zero 
    data = data[data["Quantity"] > 0]
    data = data[data["UnitPrice"] > 0]

    # strip whitespace and fill remaining nulls
    data["Description"] = data["Description"].str.strip()
    data["Description"] = data["Description"].fillna("Unknown")

    # CustomerID to int 
    data["CustomerID"] = data["CustomerID"].astype(int)

    # InvoiceDate to datetime
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])

    # total 
    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]
    return data.reset_index(drop=True)



# main
if __name__ == "__main__":
    input_file = "data/Online_Retail.xlsx"   #change if needed
    clean_file = "data/cleaned_retail.csv"

    data = load_data(input_file)
    data_clean = clean_data(data)

    data_clean.to_csv(clean_file, index=False)
