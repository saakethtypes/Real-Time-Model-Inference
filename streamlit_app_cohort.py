import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt


st.title("Cohort Analysis Bikes dataset")

# A function that will parse the date Time based cohort:  1 day of month
def get_month(x):
    x = x.split("-")
    return dt.datetime(int(x[0]), int(x[1]), 1)


@st.cache_data
def load_data():

    # Load data
    transaction_df = pd.read_csv("kpmg_transactions.csv")

    # Process data
    transaction_df = transaction_df.replace(" ", np.NaN)
    transaction_df = transaction_df.fillna(transaction_df.mean())
    transaction_df["TransactionMonth"] = transaction_df["transaction_date"].apply(
        get_month
    )
    transaction_df["TransactionYear"] = transaction_df["TransactionMonth"].dt.year
    transaction_df["TransactionMonth"] = transaction_df["TransactionMonth"].dt.month
    for col in transaction_df.columns:
        if transaction_df[col].dtype == "object":
            transaction_df[col] = transaction_df[col].fillna(
                transaction_df[col].value_counts().index[0]
            )

    # Create transaction_date column based on month and store in TransactionMonth
    transaction_df["TransactionMonth"] = transaction_df["transaction_date"].apply(
        get_month
    )
    # Grouping by customer_id and select the InvoiceMonth value
    grouping = transaction_df.groupby("customer_id")["TransactionMonth"]
    transaction_df["CohortMonth"] = grouping.transform("min")

    return transaction_df


transaction_df = load_data()

with st.expander("Show the `Bikes` dataframe"):
    st.write(transaction_df)


def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day


# Getting the integers for date parts from the `InvoiceDay` column
transcation_year, transaction_month, _ = get_date_int(
    transaction_df, "TransactionMonth"
)
# Getting the integers for date parts from the `CohortDay` column
cohort_year, cohort_month, _ = get_date_int(transaction_df, "CohortMonth")
#  Get the  difference in years
years_diff = transcation_year - cohort_year
# Calculate difference in months
months_diff = transaction_month - cohort_month

# Extract the difference in months from all previous values "+1" in addeded at the end so that first month is marked as 1 instead of 0 for easier interpretation. """
transaction_df["CohortIndex"] = years_diff * 12 + months_diff + 1

dtypes = transaction_df.dtypes.astype(str)
# Show dtypes
# dtypes

transaction_df_new_slider_01 = transaction_df[["brand", "product_line"]]
new_slider_01 = [col for col in transaction_df_new_slider_01]

transaction_df_new_slider_02 = transaction_df[["list_price", "standard_cost"]]
new_slider_02 = [col for col in transaction_df_new_slider_02]

st.write("")

cole, col1, cole, col2, cole = st.columns([0.1, 1, 0.05, 1, 0.1])

with col1:

    MetricSlider01 = st.selectbox("Pick your 1st metric", new_slider_01)

    MetricSlider02 = st.selectbox("Pick your 2nd metric", new_slider_02, index=1)

    st.write("")

with col2:

    if MetricSlider01 == "brand":
        # col_one_list = transaction_df_new["brand"].tolist()
        col_one_list = transaction_df_new_slider_01["brand"].drop_duplicates().tolist()
        multiselect = st.multiselect(
            "Select the value(s)", col_one_list, ["Solex", "Trek Bicycles"]
        )
        transaction_df = transaction_df[transaction_df["brand"].isin(multiselect)]

    elif MetricSlider01 == "product_line":
        col_one_list = (
            transaction_df_new_slider_01["product_line"].drop_duplicates().tolist()
        )
        multiselect = st.multiselect(
            "Select the value(s)", col_one_list, ["Standard", "Road"]
        )
        transaction_df = transaction_df[
            transaction_df["product_line"].isin(multiselect)
        ]

    if MetricSlider02 == "list_price":
        list_price_slider = st.slider(
            "List price (in $)", step=500, min_value=12, max_value=2091
        )
        transaction_df = transaction_df[
            transaction_df["list_price"] > list_price_slider
        ]

    elif MetricSlider02 == "standard_cost":
        standard_cost_slider = st.slider(
            "Standard cost (in $)", step=500, min_value=7, max_value=1759
        )
        transaction_df = transaction_df[
            transaction_df["list_price"] > standard_cost_slider
        ]

try:

    # Counting daily active user from each chort
    grouping = transaction_df.groupby(["CohortMonth", "CohortIndex"])
    # Counting number of unique customer Id's falling in each group of CohortMonth and CohortIndex
    cohort_data = grouping["customer_id"].apply(pd.Series.nunique)
    cohort_data = cohort_data.reset_index()
    # Assigning column names to the dataframe created above
    cohort_counts = cohort_data.pivot(
        index="CohortMonth", columns="CohortIndex", values="customer_id"
    )

    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0)
    # Coverting the retention rate into percentage and Rounding off.
    retention = retention.round(3) * 100
    retention.index = retention.index.strftime("%Y-%m")

    # Plotting the retention rate
    fig = go.Figure()

    fig.add_heatmap(
        # x=retention.columns, y=retention.index, z=retention, colorscale="cividis"
        x=retention.columns,
        y=retention.index,
        z=retention,
        # Best
        # colorscale="Aggrnyl",
        colorscale="Bluyl",
    )

    fig.update_layout(title_text="Monthly cohorts showing retention rates", title_x=0.5)
    fig.layout.xaxis.title = "Cohort Group"
    fig.layout.yaxis.title = "Cohort Period"
    fig["layout"]["title"]["font"] = dict(size=25)
    fig.layout.width = 750
    fig.layout.height = 750
    fig.layout.xaxis.tickvals = retention.columns
    fig.layout.yaxis.tickvals = retention.index
    fig.layout.plot_bgcolor = "#efefef"  # Set the background color to white
    # fig.layout.text = "num"
    fig.layout.margin.b = 100
    fig

except IndexError:
    st.warning("error")


try:

    # Counting daily active user from each chort
    grouping = transaction_df.groupby(["CohortMonth", "CohortIndex"])
    # Counting number of unique customer Id's falling in each group of CohortMonth and CohortIndex
    cohort_data = grouping["customer_id"].apply(pd.Series.nunique)
    cohort_data = cohort_data.reset_index()
    # Assigning column names to the dataframe created above
    cohort_counts = cohort_data.pivot(
        index="CohortMonth", columns="CohortIndex", values="customer_id"
    )

    cohort_sizes = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_sizes, axis=0)
    # Coverting the retention rate into percentage and Rounding off.
    retention = retention.round(3) * 100
    retention.index = retention.index.strftime("%Y-%m")

    # Plotting the retention rate
    fig = go.Figure()

    fig.add_heatmap(
        # x=retention.columns, y=retention.index, z=retention, colorscale="cividis"
        x=retention.columns,
        y=retention.index,
        z=retention,
        # Best
        # colorscale="Aggrnyl",
        colorscale="Bluyl",
    )

    fig.update_layout(title_text="Monthly cohorts showing retention rates", title_x=0.5)
    fig.layout.xaxis.title = "Cohort Group"
    fig.layout.yaxis.title = "Cohort Period"
    fig["layout"]["title"]["font"] = dict(size=25)
    fig.layout.width = 750
    fig.layout.height = 750
    fig.layout.xaxis.tickvals = retention.columns
    fig.layout.yaxis.tickvals = retention.index
    fig.layout.plot_bgcolor = "#efefef"  # Set the background color to white
    # fig.layout.text = "num"
    fig.layout.margin.b = 100
    fig

except IndexError:
    st.warning("error")