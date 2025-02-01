import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans

# Set a consistent theme for all visualizations
sns.set_theme(style="whitegrid", palette="mako")

# Function to clean data
def cleaning(df):
    df = df.drop_duplicates().dropna()  # Remove duplicates and NA values
    return df

# Function for K-means clustering
def k_means(df, clusters):
    # Aggregate data by customer
    customer_data = df.groupby('customer').agg({
        'total': 'sum',  # Total spending per customer
        'age': 'first'   # Assume age is the same for each customer
    }).reset_index()

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    customer_data['cluster'] = kmeans.fit_predict(customer_data[['total', 'age']])
    return customer_data[['customer', 'age', 'total', 'cluster']]

# Function for pie chart
def mypie(df):
    payment_totals = df.groupby('paymentType')['total'].sum()
    labels = payment_totals.index
    sizes = payment_totals.values
    colors = sns.color_palette("mako", len(labels))
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Cash vs Credit Total Spending')
    st.pyplot(fig)

# Function for box plot
def mybox(df):
    colors = sns.color_palette("mako")
    fig, ax = plt.subplots()
    sns.boxplot(y=df['total'], ax=ax, color=colors[1],saturation=0.4,gap=1.5)
    ax.set_title('Total Spending Box Plot')
    st.pyplot(fig)

# Function for scatter plot
def mypoints(df):
    age_total = df.groupby('age')['total'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(x='age', y='total', data=age_total, ax=ax, palette="mako")
    ax.set_title('Total Spending by Age')
    st.pyplot(fig)

# Function for clustring customers
def myclusters(customer_data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='age', 
        y='total', 
        hue='cluster', 
        data=customer_data, 
        palette="mako", 
        s=100, 
        edgecolor='black'
    )
    plt.title('Customer Clusters by Age and Total Spending', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Total Spending', fontsize=14)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt.gcf()) 

# Item frequency plot
def myfrequency(df_trans):
    item_freq = df_trans.sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=item_freq.values, y=item_freq.index, ax=ax, palette="mako")
    ax.set_title('Top 10 Items by Frequency')
    st.pyplot(fig)

# Function for bar chart
def mybar(df):
    city_total = df.groupby('city')['total'].sum().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x='city', y='total', data=city_total, ax=ax, palette="mako")
    ax.set_title('Total Spending by City')
    plt.xticks(rotation=45)  # Rotate city names for better readability
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Welcome to our App")
    
    st.sidebar.header("Enter The Info")
    support = st.sidebar.number_input("Minimum Support", min_value=0.001, max_value=1.0, value=0.1)
    confidence = st.sidebar.number_input("Minimum Confidence", min_value=0.001, max_value=1.0, value=0.5)
    clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=4, value=2)
    uploaded_file = st.sidebar.file_uploader("Upload Data File (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = cleaning(df)
        
        if support >= 0.001 and confidence >= 0.001:
            # Association rules
            items = df['items'].apply(lambda x: x.split(','))
            te = TransactionEncoder()
            te_ary = te.fit(items).transform(items)
            df_trans = pd.DataFrame(te_ary, columns=te.columns_)
            frequent_itemsets = apriori(df_trans, min_support=support, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
            
            # Format association rules
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            st.subheader("Association Rules")
            st.table(rules[['antecedents', 'consequents', 'support', 'confidence']])
            
            # K-means clustering
            st.subheader("K-means Clustering")
            clusters_df = k_means(df, clusters)
            st.table(clusters_df)
            
            # Visualizations
            st.subheader("Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                myclusters(clusters_df)
                mypoints(df)
                mypie(df)
            with col2:
                myfrequency(df_trans)
                mybar(df)
                mybox(df)
                

        else:
            st.error("Invalid input. The minimum Support and Confidence must be between 0.001 and 1.")

if __name__ == "__main__":
    main()