import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import base64, io

st.set_page_config(page_title="Blinkit Dashboard", layout="wide")

def logo_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

st.markdown("""
    <div style='text-align: center; margin-top: -30px; margin-bottom: -40px;'>
        <img src='data:image/png;base64,""" + logo_to_base64("download.png") + """' width='150'>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        body, .main, .block-container {
            background-color: white !important;
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("India's Last Minute App")

df = pd.read_csv("Blinkit_Dataset.csv.csv")
inventory_df = pd.read_csv("blinkit_inventoryNew.csv")

merged_df = df.merge(inventory_df, on='product_id', how='left')
merged_df.drop(columns=['reasons_if_delayed', 'Img', 'Emoji'], inplace=True)
merged_df.drop_duplicates(inplace=True)

merged_df.dropna(inplace=True)

merged_df['order_date'] = pd.to_datetime(merged_df['order_date'], errors='coerce')
merged_df['promised_delivery_time'] = pd.to_datetime(merged_df['promised_delivery_time'], errors='coerce')
merged_df['actual_delivery_time'] = pd.to_datetime(merged_df['actual_delivery_time'], errors='coerce')
merged_df['delivery_delay_minutes'] = (merged_df['actual_delivery_time'] - merged_df['promised_delivery_time']).dt.total_seconds() / 60

merged_df['delivery_speed'] = merged_df['delivery_time_minutes'].apply(lambda x: 'Fast' if x <= 10 else 'Moderate' if x <= 20 else 'Slow')
merged_df['rating_category'] = merged_df['rating'].apply(lambda r: 'High' if r >= 4 else 'Medium' if r >= 2.5 else 'Low')
merged_df['Available_Stock'] = merged_df['stock_received'] - merged_df['damaged_stock']
merged_df['order_hour'] = merged_df['order_date'].dt.hour
merged_df['order_day'] = merged_df['order_date'].dt.day_name()
merged_df['damage_rate'] = (merged_df['damaged_stock'] / merged_df['stock_received']).round(2)

blinkit_palette_10 = ['#FFCD00','#00A862', '#000000','#A8A8A8','#505050','#FFC300','#009E73','#343434','#E6B800','#007F5F']

st.sidebar.header("🔍 Apply Filters")
selected_category = st.sidebar.multiselect("Select Category", merged_df['category'].dropna().unique())
selected_segment = st.sidebar.multiselect("Select Customer Segment", merged_df['customer_segment'].dropna().unique())
selected_area = st.sidebar.multiselect("Select Area", merged_df['area'].dropna().unique())
selected_day = st.sidebar.multiselect("Select Day of Order", merged_df['order_day'].dropna().unique())
selected_rating = st.sidebar.multiselect("Select Rating Category", merged_df['rating_category'].dropna().unique())
selected_speed = st.sidebar.multiselect("Select Delivery Speed", merged_df['delivery_speed'].dropna().unique())

if selected_category:
    merged_df = merged_df[merged_df['category'].isin(selected_category)]
if selected_segment:
    merged_df = merged_df[merged_df['customer_segment'].isin(selected_segment)]
if selected_area:
    merged_df = merged_df[merged_df['area'].isin(selected_area)]
if selected_day:
    merged_df = merged_df[merged_df['order_day'].isin(selected_day)]
if selected_rating:
    merged_df = merged_df[merged_df['rating_category'].isin(selected_rating)]
if selected_speed:
    merged_df = merged_df[merged_df['delivery_speed'].isin(selected_speed)]



st.subheader("📄 Blinkit Dataset Preview")
st.dataframe(merged_df.head(10))
merged_df.dropna(inplace=True)
st.write("Filtered Data Shape:", merged_df.shape)

st.title("🛒 Blinkit Data Insights--Select a Question:-")

questions = [
    "1. Most Frequently Ordered Categories",
    "2. Revenue per Product Category",
    "3. Average Order Value per Customer Segment",
    "4. Payment Method Usage",
    "5. Avg Delivery Time by Category",
    "6. Delivery Status Distribution",
    "7. Top Areas by Delivery Delay",
    "8. Repeat Customer Count",
    "9. Average Rating by Segment",
    "10. Top 10 Most Ordered Products",
    "11. Top Rated Products by Price",
    "12. Top Areas by Order Count",
    "13. Sentiment Distribution",
    "14. Correlation Heatmap",
    "15. Top Products by Damage Rate",
    "16. Average price by product Category",
    "17. Order Count by Hour Of the day",
    "18. Top Categories recieving negative sentiment",
    "19. Price distribution by category"
]

selected_question = st.selectbox("Choose a specific question to explore:", questions)

if selected_question == questions[0]:
    st.subheader("📦 Most Frequently Ordered Categories")
    data = merged_df['category'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax,edgecolor='black')
    ax.set_xlabel("Order Count")
    ax.set_ylabel("Category")
    st.pyplot(fig)

elif selected_question == questions[1]:
    st.subheader("💰 Revenue per Product Category")
    data = merged_df.groupby('category')['price'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax,edgecolor='black')
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Category")
    st.pyplot(fig)

elif selected_question == questions[2]:
    st.subheader("📊 Avg Order Value per Segment")
    data = merged_df.groupby('customer_segment')['price'].mean()
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.barplot(x=data.index, y=data.values, palette=blinkit_palette_10, ax=ax,edgecolor='black')
    ax.set_ylabel("Average Order Value")
    ax.set_xlabel("Customer Segment")
    st.pyplot(fig)

elif selected_question == questions[3]:
    st.subheader("💳 Payment Method Usage")
    data = merged_df['payment_method'].value_counts()
    fig, ax = plt.subplots(figsize=(2,2))
    ax.pie(data.values, labels=data.index, colors=blinkit_palette_10, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4),textprops={'color':'red'})
    ax.axis('equal')
    
    st.pyplot(fig)

elif selected_question == questions[4]:
    st.subheader("⏱️ Avg Delivery Time by Category")
    data = merged_df.groupby('category')['delivery_time_minutes'].mean().sort_values().head(10)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax,edgecolor='black')
    ax.set_xlabel("Avg Delivery Time (min)")
    ax.set_ylabel("Category")
    st.pyplot(fig)

elif selected_question == questions[5]:
    st.subheader("📦 Delivery Status Distribution")
    data = merged_df['delivery_status'].value_counts()
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.pie(data.values, labels=data.index, colors=blinkit_palette_10, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4),textprops={'color':'red'})
    ax.axis('equal')
    
    st.pyplot(fig)

elif selected_question == questions[6]:
    st.subheader("📍 Top Areas by Delivery Delay")
    data = merged_df.groupby('area')['delivery_delay_minutes'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax)
    ax.set_xlabel("Avg Delivery Delay (min)")
    ax.set_ylabel("Area")
    st.pyplot(fig)

elif selected_question == questions[7]:
    st.subheader("🚚 Top Delivery Partners by Distance")
    data = merged_df.groupby('delivery_partner_id')['distance_km'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.lineplot(x=data.values, y=data.index.astype(str), marker='o', linewidth=2, color='#FFCD00', ax=ax)
    ax.set_xlabel("Avg Distance (km)")
    ax.set_ylabel("Delivery Partner ID")
    st.pyplot(fig)


elif selected_question == questions[8]:
    st.subheader("⭐ Average Rating by Segment")
    data = merged_df.groupby('customer_segment')['rating'].mean()
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.pie(data.values, labels=data.index, colors=blinkit_palette_10, autopct='%1.1f%%',textprops={'color':'red'})
    st.pyplot(fig)

elif selected_question == questions[9]:
    st.subheader("🥇 Top 10 Most Ordered Products")
    data = merged_df['product_name'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax)
    ax.set_xlabel("Order Count")
    ax.set_ylabel("Product")
    st.pyplot(fig)

elif selected_question == questions[10]:
    st.subheader("💎 Top Rated Products by Price")
    top_products = merged_df.groupby('product_name').agg({'rating': 'mean', 'price': 'mean'}).sort_values(by='rating', ascending=False).head(10)
    colors = sns.color_palette(blinkit_palette_10)
    fig, ax = plt.subplots(figsize=(4, 4))
    for (product, row), color in zip(top_products.iterrows(), colors):
        plt.scatter(row['price'], row['rating'], color=color, s=100, label=product)
    plt.xlabel('Average Price (₹)')
    plt.ylabel('Average Rating')
    plt.legend(title='Product Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

elif selected_question == questions[11]:
    st.subheader("📦 Top Areas by Order Count")
    data = merged_df['area'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax)
    ax.set_xlabel("Order Count")
    ax.set_ylabel("Area")
    st.pyplot(fig)

elif selected_question == questions[12]:
    st.subheader("😊 Sentiment Distribution of Customer Feedback")
    sentiment_counts = merged_df['sentiment'].value_counts()
    colors = ['#00A862', '#FFCD00', '#000000']
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sentiment_counts.values,
           labels=sentiment_counts.index,
           autopct='%1.1f%%',
           startangle=140,
           colors=colors,
           textprops={"color": "red"})
    st.pyplot(fig)


elif selected_question == questions[13]:
    st.subheader("📊 Correlation Matrix for Key Numerical Columns")
    corr_matrix = merged_df[['price', 'rating', 'order_total', 'delivery_time_minutes', 'distance_km']].corr()
    st.dataframe(corr_matrix.style.format("{:.2f}"))
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(corr_matrix, annot=True, cmap=blinkit_palette_10, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix: Price, Rating, Delivery Time, etc.')
    st.pyplot(fig)


elif selected_question == questions[14]:
    st.subheader("📦 Top Products by Damage Rate")
    data = merged_df.groupby('product_name')['damage_rate'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax)
    ax.set_xlabel("Damage Rate")
    ax.set_ylabel("Product")
    st.pyplot(fig)

elif selected_question == questions[15]:
    st.subheader("💰 Average Price by Product Category")
    avg_price_by_category = merged_df.groupby('category')['price'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(avg_price_by_category.index, avg_price_by_category.values, marker='o', linestyle='-', color='#00A862')
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Average Price (₹)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    st.pyplot(fig)    


elif selected_question == questions[16]:
    st.subheader("⏰ Order Count by Hour of the Day")
    merged_df['order_hour'] = pd.to_datetime(merged_df['order_date']).dt.hour
    hourly_orders = merged_df['order_hour'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(hourly_orders.index, hourly_orders.values, marker='o', linestyle='-', color='#00A862')

    ax.set_xlabel('Hour of Day (0–23)')
    ax.set_ylabel('Number of Orders')
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    st.pyplot(fig)

elif selected_question == questions[17]:
    st.subheader("😠 Top Categories Receiving Negative Sentiment")

    negative_category_counts = merged_df[merged_df['sentiment'] == 'Negative']['category'].value_counts().head(5)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.pie(negative_category_counts.values,
           labels=negative_category_counts.index,
           autopct='%1.1f%%',
           colors=sns.color_palette(blinkit_palette_10, n_colors=5),
           startangle=140,
           textprops={'color': 'red'})
    st.pyplot(fig)


elif selected_question == questions[18]:
    st.subheader("📦 Price Distribution by Product Category (Below ₹1000)")
    filtered_df = merged_df[merged_df['price'] < 1000]
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.boxplot(
        x='category',
        y='price',
        data=filtered_df,
        palette=blinkit_palette_10,
        showfliers=True,
        linewidth=1.2,
        ax=ax
    )
    ax.set_xlabel('Product Category', fontsize=12)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.markdown("*Dashboard 💻*")

