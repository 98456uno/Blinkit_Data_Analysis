import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import base64, io
import os

st.set_page_config(page_title="Blinkit Dashboard", layout="wide")

# --- Helper: load logo to base64 (safe if file exists) ---
def logo_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_b64 = logo_to_base64("download.png")

if logo_b64:
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: -30px; margin-bottom: -30px;'>
            <img src='data:image/png;base64,{logo_b64}' width='120'>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Basic style override (keeps UI readable) ---
st.markdown(
    """
    <style>
        body, .main, .block-container {
            background-color: white !important;
            color: black !important;
        }
        .stWarning {
            color: #9b3d00;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("India's Last Minute App — Blinkit Dashboard")

# --- Load data with safe fallback errors ---
@st.cache_data
def load_data():
    try:
        df1 = pd.read_csv("Blinkit_Dataset.csv.csv")
    except Exception as e:
        st.error(f"Could not load main dataset: Blinkit_Dataset.csv.csv — {e}")
        return None, None
    try:
        df2 = pd.read_csv("blinkit_inventoryNew.csv")
    except Exception as e:
        st.error(f"Could not load inventory dataset: blinkit_inventoryNew.csv — {e}")
        return None, None
    return df1, df2

df, inventory_df = load_data()
if df is None or inventory_df is None:
    st.stop()

# --- Merge + initial cleaning ---
merged_df = df.merge(inventory_df, on="product_id", how="left")

# drop columns if exist
for col in ["reasons_if_delayed", "Img", "Emoji"]:
    if col in merged_df.columns:
        merged_df.drop(columns=[col], inplace=True)

merged_df.drop_duplicates(inplace=True)
merged_df.dropna(inplace=True, how="all")

# Safely coerce datetimes
for col in ["order_date", "promised_delivery_time", "actual_delivery_time"]:
    if col in merged_df.columns:
        merged_df[col] = pd.to_datetime(merged_df[col], errors="coerce")

# safe derived columns with guard checks
if "actual_delivery_time" in merged_df.columns and "promised_delivery_time" in merged_df.columns:
    merged_df["delivery_delay_minutes"] = (
        merged_df["actual_delivery_time"] - merged_df["promised_delivery_time"]
    ).dt.total_seconds() / 60
else:
    merged_df["delivery_delay_minutes"] = pd.NA

if "delivery_time_minutes" in merged_df.columns:
    merged_df["delivery_speed"] = merged_df["delivery_time_minutes"].apply(
        lambda x: "Fast" if pd.notna(x) and x <= 10 else ("Moderate" if pd.notna(x) and x <= 20 else "Slow")
    )
else:
    merged_df["delivery_speed"] = pd.NA

if "rating" in merged_df.columns:
    merged_df["rating_category"] = merged_df["rating"].apply(lambda r: "High" if pd.notna(r) and r >= 4 else ("Medium" if pd.notna(r) and r >= 2.5 else "Low"))
else:
    merged_df["rating_category"] = pd.NA

if "stock_received" in merged_df.columns and "damaged_stock" in merged_df.columns:
    merged_df["Available_Stock"] = merged_df["stock_received"] - merged_df["damaged_stock"]
    merged_df["damage_rate"] = (merged_df["damaged_stock"] / merged_df["stock_received"]).round(2).fillna(0)
else:
    merged_df["Available_Stock"] = pd.NA
    merged_df["damage_rate"] = pd.NA

if "order_date" in merged_df.columns:
    merged_df["order_hour"] = merged_df["order_date"].dt.hour
    merged_df["order_day"] = merged_df["order_date"].dt.day_name()
else:
    merged_df["order_hour"] = pd.NA
    merged_df["order_day"] = pd.NA

# Keep an untouched copy for charts that must ignore certain filters
original_df = merged_df.copy()

# --- Palette ---
blinkit_palette_10 = ['#FFCD00','#00A862', '#000000','#A8A8A8','#505050','#FFC300','#009E73','#343434','#E6B800','#007F5F']

# --- Sidebar filters ---
st.sidebar.header("🔍 Apply Filters (some may be blocked per chart)")
# guard: if columns not exist supply empty list
def unique_vals(col):
    return [] if col not in merged_df.columns else merged_df[col].dropna().unique().tolist()

selected_category = st.sidebar.multiselect("Select Category", unique_vals('category'))
selected_segment = st.sidebar.multiselect("Select Customer Segment", unique_vals('customer_segment'))
selected_area = st.sidebar.multiselect("Select Area", unique_vals('area'))
selected_day = st.sidebar.multiselect("Select Day of Order", unique_vals('order_day'))
selected_rating = st.sidebar.multiselect("Select Rating Category", unique_vals('rating_category'))
selected_speed = st.sidebar.multiselect("Select Delivery Speed", unique_vals('delivery_speed'))

# centralize chosen filters for convenience
active_filters = {
    "category": selected_category,
    "customer_segment": selected_segment,
    "area": selected_area,
    "order_day": selected_day,
    "rating_category": selected_rating,
    "delivery_speed": selected_speed,
}

# --- Apply filters with blocked support ---
def apply_filters(df_input, blocked_fields=None):
    """Apply currently selected filters to df_input excluding any blocked_fields (list)."""
    if blocked_fields is None:
        blocked_fields = []

    df_work = df_input.copy()

    # category
    if "category" not in blocked_fields and selected_category:
        df_work = df_work[df_work['category'].isin(selected_category)]

    # customer_segment
    if "customer_segment" not in blocked_fields and selected_segment:
        df_work = df_work[df_work['customer_segment'].isin(selected_segment)]

    # area
    if "area" not in blocked_fields and selected_area:
        df_work = df_work[df_work['area'].isin(selected_area)]

    # order_day
    if "order_day" not in blocked_fields and selected_day:
        df_work = df_work[df_work['order_day'].isin(selected_day)]

    # rating_category
    if "rating_category" not in blocked_fields and selected_rating:
        df_work = df_work[df_work['rating_category'].isin(selected_rating)]

    # delivery_speed
    if "delivery_speed" not in blocked_fields and selected_speed:
        df_work = df_work[df_work['delivery_speed'].isin(selected_speed)]

    return df_work

# --- UI: dataset preview and shape ---
st.subheader("📄 Blinkit Dataset Preview")
st.dataframe(merged_df.head(10))
st.write("Full Data Shape:", merged_df.shape)

st.title("🛒 Blinkit Data Insights — Select a Question:")

questions = [
    "1. Most Frequently Ordered Categories",
    "2. Revenue per Product Category",
    "3. Average Order Value per Customer Segment",
    "4. Payment Method Usage",
    "5. Avg Delivery Time by Category",
    "6. Delivery Status Distribution",
    "7. Top Areas by Delivery Delay",
    "8. Average Rating by Segment",
    "9. Top 10 Most Ordered Products",
    "10. Top Rated Products by Price",
    "11. Top Areas by Order Count",
    "12. Sentiment Distribution",
    "13. Correlation Heatmap",
    "14. Top Products by Damage Rate",
    "15. Average Price by Product Category",
    "16. Order Count by Hour Of the Day",
    "17. Top Categories Receiving Negative Sentiment",
    "18. Price Distribution by Category (Below ₹1000)"
]

selected_question = st.selectbox("Choose a specific question to explore:", questions)

# Helper for showing a blocked filter warning
def warn_if_blocked(blocked_list):
    # check if any blocked filter is active in sidebar
    active_blocked = []
    mapping = {
        "category": selected_category,
        "customer_segment": selected_segment,
        "area": selected_area,
        "order_day": selected_day,
        "rating_category": selected_rating,
        "delivery_speed": selected_speed
    }
    for b in blocked_list:
        if mapping.get(b):
            active_blocked.append(b)
    if active_blocked:
        # human friendly names
        names = [("Customer Segment" if x=="customer_segment" else x.replace("_"," ").title()) for x in active_blocked]
        st.warning(f"⚠️ The following filters are not applicable for this visualization and will be ignored: {', '.join(names)}")

# Small default figsize choices (kept compact)
SMALL_FIG = (4.5, 3)
MED_FIG = (6, 3.5)
WIDE_FIG = (7, 4)

# --- Visualizations with blocked-filter logic per your confirmed mapping ---

# 1. Most Frequently Ordered Categories (all filters allowed)
if selected_question == questions[0]:
    st.subheader("📦 Most Frequently Ordered Categories")
    df_plot = apply_filters(merged_df, blocked_fields=[])
    data = df_plot['category'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax, edgecolor='black')
    ax.set_xlabel("Order Count")
    ax.set_ylabel("Category")
    plt.tight_layout()
    st.pyplot(fig)

# 2. Revenue per Product Category (block customer_segment, area, order_day, rating_category, delivery_speed)
elif selected_question == questions[1]:
    st.subheader("💰 Revenue per Product Category")
    blocked = ["customer_segment", "area", "order_day", "rating_category", "delivery_speed"]
    warn_if_blocked(blocked)
    df_plot = apply_filters(original_df, blocked_fields=blocked)  # use original_df but allow category filter
    # NOTE: keep category filter allowed (we didn't block 'category')
    df_plot = apply_filters(df_plot, blocked_fields=[])  # apply remaining allowed filters (category only)
    if 'price' in df_plot.columns:
        data = df_plot.groupby('category')['price'].sum().sort_values(ascending=False).head(10)
    else:
        data = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=MED_FIG)
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax, edgecolor='black')
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Category")
    plt.tight_layout()
    st.pyplot(fig)

# 3. Average Order Value per Customer Segment (block customer_segment)
elif selected_question == questions[2]:
    st.subheader("📊 Avg Order Value per Segment")
    blocked = ["customer_segment"]
    warn_if_blocked(blocked)
    df_plot = apply_filters(original_df, blocked_fields=blocked)  # ignore segment filter
    if 'price' in df_plot.columns:
        data = df_plot.groupby('customer_segment')['price'].mean().dropna()
    else:
        data = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    sns.barplot(x=data.index, y=data.values, palette=blinkit_palette_10, ax=ax, edgecolor='black')
    ax.set_ylabel("Average Order Value")
    ax.set_xlabel("Customer Segment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# 4. Payment Method Usage (block category, customer_segment, area, rating_category, delivery_speed)
elif selected_question == questions[3]:
    st.subheader("💳 Payment Method Usage")
    blocked = ["category", "customer_segment", "area", "rating_category", "delivery_speed"]
    warn_if_blocked(blocked)
    # allow order_day only
    df_plot = apply_filters(original_df, blocked_fields=blocked)
    # apply order_day if selected
    df_plot = apply_filters(df_plot, blocked_fields=[])  # ensure allowed filters applied
    if 'payment_method' in df_plot.columns:
        data = df_plot['payment_method'].value_counts()
    else:
        data = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.45))
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)

# 5. Avg Delivery Time by Category (all filters allowed)
elif selected_question == questions[4]:
    st.subheader("⏱️ Avg Delivery Time by Category")
    df_plot = apply_filters(merged_df, blocked_fields=[])
    if 'delivery_time_minutes' in df_plot.columns:
        data = df_plot.groupby('category')['delivery_time_minutes'].mean().sort_values().head(10)
    else:
        data = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=MED_FIG)
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax, edgecolor='black')
    ax.set_xlabel("Avg Delivery Time (min)")
    ax.set_ylabel("Category")
    plt.tight_layout()
    st.pyplot(fig)

# 6. Delivery Status Distribution (all filters allowed)
elif selected_question == questions[5]:
    st.subheader("📦 Delivery Status Distribution")
    df_plot = apply_filters(merged_df, blocked_fields=[])
    data = df_plot['delivery_status'].value_counts() if 'delivery_status' in df_plot.columns else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.45))
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)

# 7. Top Areas by Delivery Delay (block area)
elif selected_question == questions[6]:
    st.subheader("📍 Top Areas by Delivery Delay")
    blocked = ["area"]
    warn_if_blocked(blocked)
    df_plot = apply_filters(original_df, blocked_fields=blocked)  # ignore area filter
    # then apply others
    df_plot = apply_filters(df_plot, blocked_fields=[]) 
    if 'delivery_delay_minutes' in df_plot.columns:
        data = df_plot.groupby('area')['delivery_delay_minutes'].mean().sort_values(ascending=False).head(10)
    else:
        data = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=MED_FIG)
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax)
    ax.set_xlabel("Avg Delivery Delay (min)")
    ax.set_ylabel("Area")
    plt.tight_layout()
    st.pyplot(fig)

# 8. Average Rating by Segment (block customer_segment)
elif selected_question == questions[7]:
    st.subheader("⭐ Average Rating by Segment")
    blocked = ["customer_segment"]
    warn_if_blocked(blocked)
    df_plot = apply_filters(original_df, blocked_fields=blocked)  # ignore segment filter
    df_plot = apply_filters(df_plot, blocked_fields=[])  # apply other allowed filters
    if 'rating' in df_plot.columns:
        data = df_plot.groupby('customer_segment')['rating'].mean().dropna()
    else:
        data = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    ax.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.45))
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)

# 9. Top 10 Most Ordered Products (all filters allowed)
elif selected_question == questions[8]:
    st.subheader("🥇 Top 10 Most Ordered Products")
    df_plot = apply_filters(merged_df, blocked_fields=[])
    data = df_plot['product_name'].value_counts().head(10) if 'product_name' in df_plot.columns else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=MED_FIG)
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax)
    ax.set_xlabel("Order Count")
    ax.set_ylabel("Product")
    plt.tight_layout()
    st.pyplot(fig)

# 10. Top Rated Products by Price (all filters allowed)
elif selected_question == questions[9]:
    st.subheader("💎 Top Rated Products by Price")
    df_plot = apply_filters(merged_df, blocked_fields=[])
    if {'rating','price','product_name'}.issubset(df_plot.columns):
        top_products = df_plot.groupby('product_name').agg({'rating':'mean','price':'mean'}).dropna().sort_values(by='rating', ascending=False).head(10)
    else:
        top_products = pd.DataFrame(columns=['rating','price'])
    fig, ax = plt.subplots(figsize=WIDE_FIG)
    colors = sns.color_palette(n_colors=len(top_products)) if len(top_products)>0 else []
    for (product, row), color in zip(top_products.iterrows(), colors):
        ax.scatter(row['price'], row['rating'], color=color, s=80, label=product)
    ax.set_xlabel('Average Price (₹)')
    ax.set_ylabel('Average Rating')
    if not top_products.empty:
        ax.legend(title='Product Name', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

# 11. Top Areas by Order Count (block area)
elif selected_question == questions[10]:
    st.subheader("📦 Top Areas by Order Count")
    blocked = ["area"]
    warn_if_blocked(blocked)
    df_plot = apply_filters(original_df, blocked_fields=blocked)  # ignore area filter
    df_plot = apply_filters(df_plot, blocked_fields=[])
    data = df_plot['area'].value_counts().head(10) if 'area' in df_plot.columns else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=MED_FIG)
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax)
    ax.set_xlabel("Order Count")
    ax.set_ylabel("Area")
    plt.tight_layout()
    st.pyplot(fig)

# 12. Sentiment Distribution (block category, area, customer_segment)
elif selected_question == questions[11]:
    st.subheader("😊 Sentiment Distribution of Customer Feedback")
    blocked = ["category", "area", "customer_segment"]
    warn_if_blocked(blocked)
    df_plot = apply_filters(original_df, blocked_fields=blocked)
    df_plot = apply_filters(df_plot, blocked_fields=[])
    if 'sentiment' in df_plot.columns:
        sentiment_counts = df_plot['sentiment'].value_counts()
    else:
        sentiment_counts = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=WIDE_FIG)
    ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.45))
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)

# 13. Correlation Heatmap (block ALL filters)
elif selected_question == questions[12]:
    st.subheader("📊 Correlation Matrix for Key Numerical Columns")
    # any active filters will be ignored
    any_filters_selected = any([bool(v) for v in active_filters.values()])
    if any_filters_selected:
        st.warning("⚠️ Correlation matrix ignores sidebar filters and always uses the full dataset.")
    # choose key numerics if exist
    cols = [c for c in ['price','rating','order_total','delivery_time_minutes','distance_km'] if c in original_df.columns]
    if cols:
        corr_matrix = original_df[cols].corr()
        st.dataframe(corr_matrix.style.format("{:.2f}"))
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for correlation.")

# 14. Top Products by Damage Rate (all filters allowed)
elif selected_question == questions[13]:
    st.subheader("📦 Top Products by Damage Rate")
    df_plot = apply_filters(merged_df, blocked_fields=[])
    if 'damage_rate' in df_plot.columns:
        data = df_plot.groupby('product_name')['damage_rate'].mean().sort_values(ascending=False).head(10)
    else:
        data = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=MED_FIG)
    sns.barplot(x=data.values, y=data.index, palette=blinkit_palette_10, ax=ax)
    ax.set_xlabel("Damage Rate")
    ax.set_ylabel("Product")
    plt.tight_layout()
    st.pyplot(fig)

# 15. Average Price by Product Category (all filters allowed)
elif selected_question == questions[14]:
    st.subheader("💰 Average Price by Product Category")
    df_plot = apply_filters(merged_df, blocked_fields=[])
    if 'price' in df_plot.columns:
        avg_price_by_category = df_plot.groupby('category')['price'].mean().sort_values(ascending=False)
    else:
        avg_price_by_category = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=WIDE_FIG)
    ax.plot(avg_price_by_category.index, avg_price_by_category.values, marker='o', linestyle='-', linewidth=1)
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Average Price (₹)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

# 16. Order Count by Hour Of the Day (all filters allowed)
elif selected_question == questions[15]:
    st.subheader("⏰ Order Count by Hour of the Day")
    df_plot = apply_filters(merged_df, blocked_fields=[])
    if 'order_hour' in df_plot.columns:
        hourly_orders = df_plot['order_hour'].value_counts().sort_index()
    else:
        hourly_orders = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=WIDE_FIG)
    ax.plot(hourly_orders.index, hourly_orders.values, marker='o', linestyle='-')
    ax.set_xlabel('Hour of Day (0–23)')
    ax.set_ylabel('Number of Orders')
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

# 17. Top Categories Receiving Negative Sentiment (block customer_segment, area, rating_category)
elif selected_question == questions[16]:
    st.subheader("😠 Top Categories Receiving Negative Sentiment")
    blocked = ["customer_segment", "area", "rating_category"]
    warn_if_blocked(blocked)
    df_plot = apply_filters(original_df, blocked_fields=blocked)
    df_plot = apply_filters(df_plot, blocked_fields=[])
    if 'sentiment' in df_plot.columns and 'category' in df_plot.columns:
        negative_category_counts = df_plot[df_plot['sentiment']=='Negative']['category'].value_counts().head(5)
    else:
        negative_category_counts = pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    ax.pie(negative_category_counts.values, labels=negative_category_counts.index, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.45))
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)

# 18. Price Distribution by Category (Below ₹1000, all filters allowed)
elif selected_question == questions[17]:
    st.subheader("📦 Price Distribution by Product Category (Below ₹1000)")
    df_plot = apply_filters(merged_df, blocked_fields=[])
    if 'price' in df_plot.columns and 'category' in df_plot.columns:
        filtered_df = df_plot[df_plot['price'] < 1000]
        fig, ax = plt.subplots(figsize=WIDE_FIG)
        sns.boxplot(x='category', y='price', data=filtered_df, palette=blinkit_palette_10, showfliers=True, linewidth=1.0, ax=ax)
        ax.set_xlabel('Product Category', fontsize=10)
        ax.set_ylabel('Price (₹)', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Price or category column not available in dataset.")

st.markdown("---")
st.markdown("*Dashboard 💻*")
