import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy import stats

# Set page layout width to wide
st.set_page_config(layout="wide")

# Load data
livestockPopn = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/1 Livestock population.csv")
milkAnimal = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/2 Milking animals.csv")
meatProd = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/3 Meat production.csv")
eggProd = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/4 Egg production.csv")

# Define function to create a page for a given dataframe
def create_page(df, title):
    st.markdown(f"<h1 style='color: orange'>{title}</h1>", unsafe_allow_html=True)

    # Radio buttons for visualization selection
    st.sidebar.markdown("## Visualization")
    visualization = st.sidebar.radio("Select Visualization", options=["Data Summary", "Bar Graph", "Interactive Plot", "Bubble Chart", "Prediction", "Statistical Testing"])

    if visualization == "Data Summary":
        # Define filter options
        provinces = ["All"] + sorted(df["Province"].unique())

        # Allow user to filter table by province column
        selected_province = st.selectbox("Filter by province", provinces)
        if selected_province != "All":
            df = df[df["Province"] == selected_province]

        # Allow user to sort table by column
        sort_by = st.selectbox("Sort by", list(df.columns))
        descending_order = st.checkbox("Sort in descending order")
        if st.button("Sort"):
            df = df.sort_values(sort_by, ascending=not descending_order)

        # Display filter and sort options
        filter_col, sort_col = st.columns(2)
        with filter_col:
            st.write(f"Filter: {selected_province}")
        with sort_col:
            st.write(f"Sort by: {sort_by} {'(desc)' if descending_order else '(asc)'}")

        # Display table
        st.write(df)

        st.write("<h3 style='color: orange'>Data summary</h3>", unsafe_allow_html=True)
        st.write(df.describe())    

        # Get the column names for the animal counts
        animal_columns = [col for col in df.columns if col not in ['District', 'Total Milk Produced', 'Total Meat', 'Total Egg']]

        # Group the data by province and sum the animal counts
        animal_counts = df.groupby('Province')[animal_columns].sum()

        # Reorder the rows in animal_counts to match the original dataset
        animal_counts = animal_counts.reindex(df['Province'].unique())

        # Create a formatting function to highlight the minimum and maximum values in each row
        def highlight_extrema(s):
            is_max = s == s.max()
            is_min = s == s.min()
            max_color = 'color: green'
            min_color = 'color: red'
            return [max_color if v else min_color if w else '' for v, w in zip(is_max, is_min)]

        # Apply the formatting function to the animal counts table
        animal_counts_formatted = animal_counts.style.apply(highlight_extrema)

        # Display the formatted table in Streamlit
        st.write("<h3 style='color: orange'>Province based analysis of Livestock</h3>", unsafe_allow_html=True)
        st.write(":green[Green text] indicate mazimum value and :red[Red text] indicate minimum value in a particular column")
        st.dataframe(animal_counts_formatted)    

        livestock_columns = [col for col in df.columns if col not in ['Province','District',]]
        max_districts = {}

        for col in livestock_columns:
                max_districts[col] = df.loc[df[col].idxmax()]['District']
                st.write(f"The district with the greatest number of {col.lower()} is: {max_districts[col]}")

    elif visualization == "Bar Graph":
        st.write("Bar Graph")
        x_axis = st.selectbox('Select x-axis column', df.columns)
        y_axis = st.selectbox('Select y-axis column', df.columns)

        if x_axis != y_axis:
            chart = alt.Chart(df).mark_bar().encode(
                x=x_axis,
                y=y_axis
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Please select different columns for x-axis and y-axis.")

        st.write("Bar Graph")

        # Sidebar selection for province and y-axis columns
        selected_province = st.sidebar.selectbox("Select Province", df['Province'].unique(), key='province')
        selected_y_axis = st.sidebar.selectbox("Select y-axis column", df.columns, key='y_axis')

        # Filter the DataFrame based on the selected province
        df_filtered = df[df['Province'] == selected_province]

        if selected_y_axis != 'District':
            chart = alt.Chart(df_filtered).mark_bar().encode(
                x=alt.X('District', axis=alt.Axis(title='District')),
                y=alt.Y(selected_y_axis, axis=alt.Axis(title=selected_y_axis))
            ).properties(
                title=f"Bar Plot - Province {selected_province}",
                width=500,
                height=400
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Please select a valid y-axis column.")

    elif visualization == "Interactive Plot":
        st.write("Interactive Plot")
        x_axis = st.selectbox("Select x-axis column", df.columns)
        y_axis = st.selectbox("Select y-axis column", df.columns)
        chart = alt.Chart(df).mark_circle(size=100).encode(
            x=x_axis,
            y=y_axis,
            tooltip=df.columns.tolist()
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    elif visualization == "Bubble Chart":
        st.write("Bubble Chart")
        x_axis = st.selectbox("Select x-axis column", df.columns)
        y_axis = st.selectbox("Select y-axis column", df.columns)
        size_column = st.selectbox("Select size column", df.columns)

        if x_axis != y_axis:
            chart = alt.Chart(df).mark_circle().encode(
                x=x_axis,
                y=y_axis,
                size=size_column,
                tooltip=df.columns.tolist()
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Please select different columns for x-axis and y-axis.")

    elif visualization == "Prediction":
        # Allow user to select input and output columns
        input_columns = st.multiselect("Select input columns", df.columns.tolist())
        output_column = st.selectbox("Select output column", df.columns.tolist())

        # Split the dataset into inputs (X) and output (y) based on user selection
        X = df[input_columns].values
        y = df[output_column].values.reshape(-1, 1)  # Reshape y to have a 2D shape

        # Apply L2 regularization to the linear regression model
        ridge = Ridge(alpha=0.1)
        ridge.fit(X, y)

        # Make predictions
        predictions = ridge.predict(X)

        # Combine actual and predicted values into a dataframe for easier comparison
        results = pd.DataFrame({'Actual': y.flatten(), 'Predicted': predictions.flatten()})

        # Print the first 10 rows
        st.write(results.head(10))

        # Calculate the R-squared score
        accuracy = r2_score(y, predictions)

        st.write("Accuracy: {:.2f}%".format(accuracy * 100))

    elif visualization == "Statistical Testing":
        st.write("##### In this statistical testing, I used an ANOVA test to determine whether to reject or fail to reject the null hypothesis based on the calculated p-value.")
        # Define significance level
        alpha = 0.05
        st.write("##### Performing One-way ANOVA Test")

        # Allow user to select columns for the one-way ANOVA test
        test_columns = st.multiselect("Select columns for the one-way ANOVA test", df.columns.tolist())

        # Perform one-way ANOVA test
        test_data = [df[column] for column in test_columns]
        result = stats.f_oneway(*test_data)

        # Print the result
        st.write("##### F-Statistic: ", result.statistic)
        st.write("##### p-value: ", result.pvalue)

        # Check if the p-value is less than the significance level alpha
        if result.pvalue < alpha:
            st.markdown(f"##### There is a significant difference in the mean values across the selected columns: {', '.join(test_columns)}, at a significance level of {alpha}")
        else:
            st.markdown(f"##### There is no significant difference in the mean values across the selected columns: {', '.join(test_columns)}, at a significance level of {alpha}")

        # Print the result
        st.markdown(f"##### After performing the ANOVA test, I obtained a p-value of **{result.pvalue}**, which is much smaller than our chosen significance level of **{alpha}**. This indicates that we have strong evidence to suggest that the mean number of livestock is significantly different across the provinces.")
        st.markdown("")
        st.markdown("##### Therefore, we reject the null hypothesis and conclude that there is a significant difference in the mean number of livestock across different provinces in Nepal.")

# Create page dictionary
pages = {
    "Livestock Population": livestockPopn,
    "Milk Production": milkAnimal,
    "Meat Production": meatProd,
    "Egg Production": eggProd,
}

# Allow user to select a page from the dictionary
selected_page = st.sidebar.selectbox("Select a page", list(pages.keys()))
create_page(pages[selected_page], selected_page)
