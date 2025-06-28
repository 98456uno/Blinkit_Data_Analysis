#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("/content/Blinkit_Dataset.csv.csv")
df


# In[ ]:


inventory_df = pd.read_csv("/content/blinkit_inventoryNew.csv")


# In[ ]:


merged_df = df.merge(inventory_df, on='product_id', how='left')
merged_df


# In[ ]:


merged_df.info()


# In[ ]:


merged_df.describe()


# In[ ]:


merged_df.isnull().sum()


# In[ ]:


merged_df.dropna
merged_df.shape


# In[ ]:


merged_df.drop(columns=['reasons_if_delayed'], inplace=True)


# In[ ]:


merged_df.isnull().sum()


# In[ ]:


merged_df['order_id'].duplicated().sum()


# In[ ]:


merged_df_unique_orders = df.drop_duplicates(subset='order_id', keep='first')


# In[ ]:


merged_df.shape


# In[ ]:


merged_df.drop_duplicates(inplace=True)


# In[ ]:


merged_df.shape


# In[ ]:


merged_df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
merged_df['promised_delivery_time'] = pd.to_datetime(df['promised_delivery_time'], errors='coerce')
merged_df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'], errors='coerce')


# In[ ]:


def categorize_speed(minutes):
    if minutes <= 10:
        return 'Fast'
    elif minutes <= 20:
        return 'Moderate'
    else:
        return 'Slow'

merged_df['delivery_speed'] = merged_df['delivery_time_minutes'].apply(categorize_speed)


# In[ ]:


merged_df


# In[ ]:


def rate_bucket(r):
    if r >= 4:
        return 'High'
    elif r >= 2.5:
        return 'Medium'
    else:
        return 'Low'

merged_df['rating_category'] = merged_df['rating'].apply(rate_bucket)


# In[ ]:


merged_df


# In[ ]:


merged_df['Available_Stock'] = merged_df['stock_received'] - merged_df['damaged_stock']
merged_df.dropna()
df.shape


# In[ ]:


print(merged_df.columns)


# In[ ]:


merged_df.drop(columns=['Img','Emoji'], inplace=True)


# In[ ]:


print(merged_df.columns)


# **Order Level Insights**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
print("Most Frequently Ordered Product Categories:")
category_counts = merged_df['category'].value_counts()
print(category_counts)

blinkit_palette_10 = [ '#FFCD00','#00A862', '#000000','#A8A8A8',  '#505050',  '#FFC300', '#009E73',  '#343434',  '#E6B800', '#007F5F']


plt.figure(figsize=(10,6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette=blinkit_palette_10,edgecolor='black')
plt.title('Most Frequently Ordered Product Categories')
plt.xlabel('Category')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# In[ ]:


print("Total Revenue per Product Category:")
Total_Revenue=merged_df.groupby('category')
Total_Revenue = merged_df.groupby('category')['order_total'].sum().sort_values(ascending=False) / 1e5
print(Total_Revenue)


plt.figure(figsize=(10,6))
sns.barplot(y=Total_Revenue.index, x=Total_Revenue.values, palette=blinkit_palette_10,edgecolor='black')
plt.title('Total Revenue per Product Category')
plt.xlabel('Revenue (₹ Lakh)')
plt.ylabel('Product Category')
plt.tight_layout()
plt.show()


# In[ ]:


print("Average order value per customer segment")
Average_Order_Value = merged_df.groupby("customer_segment")
Average_Order_Value = Average_Order_Value["order_total"].mean().sort_values(ascending=False)
print(Average_Order_Value)

plt.figure(figsize=(13,4))
sns.barplot(x=Average_Order_Value.values, y=Average_Order_Value.index, palette=blinkit_palette_10,edgecolor='black')
plt.title('Average Order Value per Customer Segment')
plt.xlabel('Average Order Value (₹)')
plt.ylabel('Customer Segment')

for i, v in enumerate(Average_Order_Value.values):
    plt.text(v + 10, i, f'{v:.2f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()


# 

# In[ ]:


print("payment meathod usage frequency")
payment_method_counts = merged_df['payment_method'].value_counts()
print("Payment methods usage frequency")
print(payment_method_counts)

plt.figure(figsize=(7,7))
plt.pie( payment_method_counts.values, labels=payment_method_counts.index, autopct='%1.1f%%',startangle=90,colors=sns.color_palette(blinkit_palette_10),wedgeprops={'width': 0.4},textprops={'color':'red'})
plt.title('Payment Method Usage Frequency')
plt.tight_layout()
plt.show()


# **DELIVERY** & **LOGISTICS**

# In[ ]:


print("Average Delivery Time per Category:")
Delivery_time_category = merged_df.groupby('category')
Delivery_time_category=Delivery_time_category['delivery_time_minutes'].mean().sort_values()
print(Delivery_time_category)

plt.figure(figsize=(10,6))
sns.barplot(x=Delivery_time_category.values, y=Delivery_time_category.index, palette=blinkit_palette_10,edgecolor='black')
plt.title('Average Delivery Time per Category')
plt.xlabel('Average Delivery Time (minutes)')
plt.ylabel('Product Category')


plt.tight_layout()
plt.show()


# In[ ]:


print("Delivery Status Distribution (%):")
delivery_percentage = merged_df['delivery_status'].value_counts(normalize=True) * 100
print(delivery_percentage)

import matplotlib.pyplot as plt

delivery_percentage = merged_df['delivery_status'].value_counts(normalize=True) * 100

plt.figure(figsize=(7,7))
plt.pie(delivery_percentage.values,labels=delivery_percentage.index,autopct='%1.1f%%',startangle=90,colors=sns.color_palette(blinkit_palette_10),wedgeprops={'width': 0.4},textprops={'color': 'Red'})
plt.title('Delivery Status Distribution (%)')
plt.tight_layout()
plt.show()


# In[ ]:


merged_df['delivery_delay_minutes'] = (pd.to_datetime(merged_df['actual_delivery_time']) - pd.to_datetime(merged_df['promised_delivery_time'])).dt.total_seconds() / 60
print("Top 4 Areas with Highest Average Delivery Delay (in minutes):")
delay_by_area=merged_df.groupby('area')['delivery_delay_minutes'].mean().dropna().sort_values(ascending=False)
print(delay_by_area.head(4))

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x=delay_by_area.values, y=delay_by_area.index, palette=blinkit_palette_10,edgecolor='black')
plt.title('Top  Areas with Highest Average Delivery Delay (in Minutes)')
plt.xlabel('Average Delay (minutes)')
plt.ylabel('Area')

plt.tight_layout()
plt.show()


# In[ ]:


print("Average Distance Covered per Delivery Partner:")

avg_distance = merged_df.groupby('delivery_partner_id')['distance_km'].mean().sort_values(ascending=False).head(10)
print(avg_distance)

plt.figure(figsize=(12,6))
sns.lineplot(x=avg_distance.index, y=avg_distance.values, marker='o', linewidth=2, color='yellow')
plt.title('Top 10 Delivery Partners by Average Distance Covered')
plt.xlabel('Delivery Partner ID')
plt.ylabel('Average Distance (km)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()





# **CUSTOMER BEHAVIOUR & SEGMENTATION**

# In[ ]:


print("Top 10 Customers by Total Order Value:")
customer = merged_df.groupby('customer_id')['order_total'].sum().sort_values(ascending=False).head(10)
print(customer)


plt.figure(figsize=(10, 6))
sns.barplot(
    x=customer.index,
    y=customer.values,
    palette=blinkit_palette_10,
    edgecolor='black'
)

plt.title('Top 10 Customers by Total Order Value', fontsize=14)
plt.xlabel('Customer_ID', fontsize=12)
plt.ylabel('Total Order Value', fontsize=12)
plt.tight_layout()
plt.show()


# In[ ]:


repeat_counts = merged_df['customer_id'].value_counts()
print("Number of Repeat Customers:")
print((repeat_counts > 1).sum())


# In[ ]:


import matplotlib.pyplot as plt

avg_rating_segment = merged_df.groupby('customer_segment')['rating'].mean().sort_values(ascending=False)

plt.figure(figsize=(8,8))
plt.pie(
    avg_rating_segment.values,
    labels=avg_rating_segment.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette(blinkit_palette_10),
    textprops={'color': 'Red'}
)
plt.title('Average Rating by Customer Segment (Proportional View)')
plt.tight_layout()
plt.show()


# **PRODUCT LEVEL INSIGHTS**

# In[ ]:


print(" Top 10 Most Ordered Products:")
top_products = merged_df['product_name'].value_counts().head(10)
print(top_products)

plt.figure(figsize=(10,6))
sns.barplot(x=top_products.values, y=top_products.index, palette=blinkit_palette_10,edgecolor='black')
plt.title('Top 10 Most Ordered Products')
plt.xlabel('Number of Orders')
plt.ylabel('Product Name')

plt.tight_layout()
plt.show()



# In[ ]:


top_products = merged_df.groupby('product_name').agg({'rating': 'mean', 'price': 'mean'}).sort_values(by='rating', ascending=False).head(10)
print("Product-wise Average Rating and Price:")
print(top_products)
colors = sns.color_palette(blinkit_palette_10)

plt.figure(figsize=(10,6))
for (product, row), color in zip(top_products.iterrows(), colors):
    plt.scatter(row['price'], row['rating'], color=color, s=100, label=product)

plt.title('Top 10 Products by Rating: Rating vs Price')
plt.xlabel('Average Price (₹)')
plt.ylabel('Average Rating')
plt.grid(True, linestyle='--', alpha=0.5)

plt.legend(title='Product Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# In[ ]:


import matplotlib.pyplot as plt

avg_price_by_category = merged_df.groupby('category')['price'].mean().sort_values(ascending=False)
print(avg_price_by_category)

plt.figure(figsize=(10,6))
plt.plot(avg_price_by_category.index, avg_price_by_category.values, marker='o', linestyle='-', color='#00A862')

plt.title('Average Price by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average Price (₹)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# **AREA & GEOGRAPHIC INSIGHTS**

# In[ ]:


print("Top 10 Areas by Order Count:")
top_areas_order_count=merged_df['area'].value_counts().head(10)
print(top_areas_order_count)

plt.figure(figsize=(10,6))
sns.barplot(x=top_areas_order_count.index, y=top_areas_order_count.values, palette=blinkit_palette_10,edgecolor='black')

plt.title('Top 10 Areas by Order Count')
plt.xlabel('Area')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# In[ ]:


top_rated_areas = merged_df.groupby('area')['rating'].mean().sort_values(ascending=False).head(20)
print("Top 20 Areas by Average Rating:")
print(top_rated_areas)

plt.figure(figsize=(10,6))
plt.plot(top_rated_areas.values, top_rated_areas.index, marker='o', linestyle='-', color='#00A862')

plt.title('Top 10 Areas by Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Area')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# In[ ]:


merged_df['order_hour'] = pd.to_datetime(merged_df['order_date']).dt.hour
print("Order Count by Hour of the Day:")
hourly_orders = merged_df['order_hour'].value_counts().sort_index()
print("Order Count by Hour of the Day:")
print(hourly_orders)

plt.figure(figsize=(10,6))
plt.plot(hourly_orders.index, hourly_orders.values, marker='o', linestyle='-', color='#00A862')

plt.title('Order Count by Hour of the Day')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Number of Orders')
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# In[ ]:


merged_df['order_day'] = pd.to_datetime(merged_df['order_date']).dt.day_name()
print("Order Count by Day of the Week:")
order_day = merged_df['order_day'].value_counts()
print(order_day)

plt.figure(figsize=(10,6))
plt.plot(order_day.index, order_day.values, marker='o', linestyle='-', color='#FFCD00')

plt.title('Order Count by Day of the Week')
plt.xlabel('Day')
plt.ylabel('Number of Orders')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



# *FEEDBACK & SENTIMENT ANALYSIS*

# In[ ]:


print("Sentiment Distribution of Customer Feedback:")
sentiment_counts = merged_df['sentiment'].value_counts()
print("Sentiment Distribution of Customer Feedback:")
print(sentiment_counts)

colors = ['#00A862', '#FFCD00', '#000000']

plt.figure(figsize=(6,6))
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=colors,textprops={"color":"red"})

plt.title('Sentiment Distribution of Customer Feedback')
plt.tight_layout()
plt.show()



# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sentiment_rating = merged_df.groupby('sentiment')['rating'].mean()
print("Average Rating per Sentiment Category:")
print(sentiment_rating)

plt.figure(figsize=(8,5))
sns.barplot(x=sentiment_rating.index, y=sentiment_rating.values, palette=blinkit_palette_10,edgecolor='black')


plt.title('Average Rating per Sentiment Category')
plt.xlabel('Sentiment')
plt.ylabel('Average Rating')
plt.ylim(0, 5.5)
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# In[ ]:


print("Top Categories Receiving Negative Sentiment:")
print("Unique Sentiment Values:")
print(df['sentiment'].unique())

negative_category_counts=merged_df[merged_df['sentiment'] == 'Negative']['category'].value_counts().head(5)

plt.figure(figsize=(6,6))
plt.pie(negative_category_counts.values, labels=negative_category_counts.index,
        autopct='%1.1f%%', colors=sns.color_palette(blinkit_palette_10, n_colors=5), startangle=140,textprops={'color':'red'})
plt.title('Negative Sentiment Share by Category')
plt.tight_layout()
plt.show()



# In[ ]:


print("Correlation Matrix for Numerical Columns:")
corr_matrix=(merged_df[['price', 'rating', 'order_total', 'delivery_time_minutes', 'distance_km']].corr())
print(corr_matrix)


plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap=blinkit_palette_10, fmt=".2f", linewidths=0.5)

plt.title('Correlation Matrix: Price, Rating, Delivery Time, etc.')
plt.tight_layout()
plt.show()



# In[ ]:


print("Covariance Between Price and Rating:")
corr_matrix=(merged_df[['price', 'rating']].cov())
print(corr_matrix)
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=blinkit_palette_10, linewidths=0.5, linecolor='white')

plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()



# In[ ]:


merged_df['damage_rate'] = (merged_df['damaged_stock'] / merged_df['stock_received']).round(2)

plt.figure(figsize=(9,6))
sns.histplot(data=merged_df, x='damage_rate', bins=20, color='#FFCD00', edgecolor='black')

plt.title('Distribution of Damage Rate')
plt.xlabel('Damage Rate')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[ ]:


print("Top 10 Products with Highest Damage Rate")
top_damaged_products = merged_df.groupby('product_name')['damage_rate'].mean().sort_values(ascending=False).head(10)
print(top_damaged_products)
plt.figure(figsize=(10,6))
sns.barplot(x=top_damaged_products.values, y=top_damaged_products.index, palette=blinkit_palette_10,edgecolor='black')

plt.title('Top 10 Products with Highest Average Damage Rate')
plt.xlabel('Average Damage Rate')
plt.ylabel('Product Name')
plt.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()



# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

filtered_df = merged_df[merged_df['price'] < 1000]

plt.figure(figsize=(12, 6))
sns.boxplot(
 x='category',
    y='price',
    data=filtered_df,
    palette=blinkit_palette_10,
    showfliers=True,
    linewidth=1.2
)

plt.title('Price Distribution by Product Category', fontsize=14, weight='bold')
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Price (₹)', fontsize=12)
plt.xticks(rotation=45, ha='right')  # for readability
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

top_categories = df['category'].value_counts().nlargest(5).index
cat_stats = merged_df.groupby('category').agg({
    'order_id': 'count',
    'order_total': 'sum',
    'rating': 'mean'
}).loc[top_categories]

plt.figure(figsize=(10, 6))
plt.bar(cat_stats.index, cat_stats['order_id'], color="#FFD700", label='Orders')
plt.plot(cat_stats.index, cat_stats['order_total'] / 1000, color="#7AC142", marker='o', label='Revenue (K)')
plt.plot(cat_stats.index, cat_stats['rating'] * 1000, color="#000000", marker='x', label='Avg Rating x1000')
plt.title("Orders, Revenue & Avg Rating by Category")
plt.legend()
plt.ylabel("Count / Rating / Revenue")
plt.tight_layout()
plt.show()


# In[ ]:




