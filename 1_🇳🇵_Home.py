import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from sklearn.cluster import KMeans
from datetime import datetime
import pytz

# Set page title and favicon
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: orange;'>Livestock Data Analysis of Nepal</h1>", unsafe_allow_html=True)

# Function to get the current time in Nepal
def get_nepal_time():
    nepal_timezone = pytz.timezone('Asia/Kathmandu')
    current_time = datetime.now(nepal_timezone)
    return current_time.strftime("%H:%M:%S")

current_time = get_nepal_time()
st.markdown(f"<h5 style='text-align: left; color: orange;'>Current time in Nepal: {current_time}</h5>", unsafe_allow_html=True)
# Display the current time in Nepal

# Functions for each of the pagesi
def home(uploaded_file):
    if uploaded_file:
        st.write(f"Data shape: {df.shape}")

        st.write("Data information:")
        data = {
        'Column Name': df.columns,
        'Data Type': df.dtypes.values,
        'Missing Values': df.isnull().sum().values,
        'Total Rows': len(df)
        }
        df_info = pd.DataFrame(data)
        # display the new dataframe
        st.dataframe(df_info)

        # Get list of columns in the dataframe
        columns = df.columns.tolist()

        # Allow user to select a column for filtering
        selected_column = st.selectbox("Filter by column", columns)
        column_values = ["All"] + sorted(df[selected_column].unique())

        # Allow user to select a value to filter by
        selected_value = st.selectbox(f"Filter by {selected_column}", column_values)
        if selected_value != "All":
            df_filtered = df[df[selected_column] == selected_value]
        else:
            df_filtered = df

        # Allow user to sort table by column
        sort_by = st.selectbox("Sort by", columns)
        descending_order = st.checkbox("Sort in descending order")
        if st.button("Sort"):
            df_filtered = df_filtered.sort_values(sort_by, ascending=not descending_order)

        # Display filter and sort options
        filter_col, sort_col = st.columns(2)
        with filter_col:
            st.write(f"Filter: {selected_column} - {selected_value}")
        with sort_col:
            st.write(f"Sort by: {sort_by} {'(desc)' if descending_order else '(asc)'}")

        # Display table
        st.write(df_filtered)
        
    else:
        st.markdown("<h3 style='font-size: 23px; color: orange;'>To begin, please upload a CSV file</h3>", unsafe_allow_html=True)


def data_summary():
    st.header('Statistics of Dataframe')
    st.write(df.describe())

    st.header('Header of Dataframe')
    st.write(df.head())

    st.header('Correlation Matrix')
    corr_matrix = df.corr()
    st.write(corr_matrix)

    st.header('Boxplot')
    cols = st.multiselect('Select columns to plot', options=df.columns)
    if cols:
        fig, ax = plt.subplots()
        df[cols].boxplot(ax=ax)
        st.pyplot(fig)

    st.header('Histogram')
    col = st.selectbox('Select column', options=df.columns)
    if col:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

def displayplot():
    st.header('Interactive Bar Graph')
    
    x_axis = st.selectbox('Select x-axis column', df.columns)
    y_axis = st.selectbox('Select y-axis column', df.columns)
    
    if x_axis in df.columns and y_axis in df.columns:
        chart = alt.Chart(df).mark_bar().encode(
            x=x_axis,
            y=y_axis
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)


def interactive_plot():
    col1, col2 = st.columns(2)
    
    x_axis_val = col1.selectbox('Select the X-axis', options=df.columns)
    y_axis_val = col2.selectbox('Select the Y-axis', options=df.columns)

    plot = px.scatter(df, x=x_axis_val, y=y_axis_val)
    # Increase the symbol size
    plot.update_traces(marker=dict(size=10))
    st.plotly_chart(plot, use_container_width=True)

def cluster_data():
    st.header('Cluster Data')

    # Get the columns for clustering
    cluster_columns = st.multiselect('Select columns for clustering', options=df.columns)

    num_clusters = st.slider('Number of Clusters', min_value=2, max_value=10, value=2, step=1)
    
    # Subset the dataset with selected columns
    df_subset = df[cluster_columns].copy()

    # Preprocess the dataset by encoding categorical variables
    df_encoded = pd.get_dummies(df_subset)
    
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df_encoded)
    
    df_subset['Cluster'] = kmeans.labels_
    st.write(df_subset)
    
    # Plot the clusters
    fig = px.scatter(df_subset, x=cluster_columns[0], y=cluster_columns[1], color='Cluster', color_continuous_scale='RdYlGn')
    # Increase the symbol size
    fig.update_traces(marker=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)


# Sidebar setup
st.sidebar.title('Sidebar')
upload_file = st.sidebar.file_uploader('Upload a file containing livestock data')
#Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', ['Home', 'Data Summary', 'Bar Graph', 'Interactive Plots', 'Cluster Data'])

# Check if file has been uploaded
if upload_file is not None:
    df = pd.read_csv(upload_file)

# Navigation options
if options == 'Home':
    home(upload_file)
elif options == 'Data Summary':
    data_summary()
elif options == 'Bar Graph':
    displayplot()
elif options == 'Interactive Plots':
    interactive_plot()
elif options == 'Cluster Data':
    cluster_data()
