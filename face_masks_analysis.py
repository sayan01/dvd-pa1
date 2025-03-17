"""
Face Masks Exploratory Data Analysis and Visualization

This script performs EDA and visualization on face mask products and reviews
to provide insights to a manufacturer of personal care products.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter
import os
import string

# Set visualization style
plt.style.use('ggplot')
sns.set_palette('viridis')

# Configure plot settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Load the data
print("Loading data...")
products_df = pd.read_csv('products.tsv', sep='\t')
reviews_df = pd.read_csv('reviews.tsv', sep='\t')

print(f"Products dataset shape: {products_df.shape}")
print(f"Reviews dataset shape: {reviews_df.shape}")

# Data Cleaning and Preparation
print("\nCleaning and preparing data...")

# Clean the product price column
products_df['product_price'] = pd.to_numeric(products_df['product_price'], errors='coerce')

# Create a standardized rating column in reviews
# The ratingValue seems to be on a 10-50 scale (10=1 star, 20=2 stars, etc.)
reviews_df['rating_stars'] = reviews_df['ratingValue'] / 10

# Convert postedDate to datetime
reviews_df['postedDate'] = pd.to_datetime(reviews_df['postedDate'], errors='coerce')

# Extract product type from product name
def extract_mask_type(name):
    if not isinstance(name, str):
        return 'Unknown'
    
    name = name.lower()
    if any(k in name for k in ['kn95', 'n95', 'ffp2', 'kf94', 'respirator']):
        return 'Respirator'
    elif any(k in name for k in ['cotton', 'reusable', 'cloth']):
        return 'Cloth/Reusable'
    elif 'disposable' in name:
        return 'Disposable'
    elif 'copper' in name:
        return 'Copper'
    elif 'nano' in name:
        return 'Nano Technology'
    elif 'filter' in name and 'pocket' not in name:  # Filter as primary product, not just a feature
        return 'Filter'
    elif 'kid' in name:
        return 'Kids'
    else:
        return 'Other'

products_df['mask_type'] = products_df['product_name'].apply(extract_mask_type)

# Extract other product features
def extract_pack_size(name):
    if not isinstance(name, str):
        return 1
        
    # Try to find patterns like "10 Pack", "5 Count", etc.
    patterns = [r'(\d+)\s+Pack', r'(\d+)\s+Count', r'(\d+)\s+Masks?', r'(\d+)\s+Mask']
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    # Default to 1 if no pack size found
    return 1

products_df['pack_size'] = products_df['product_name'].apply(extract_pack_size)

# Calculate price per mask
products_df['price_per_mask'] = products_df['product_price'] / products_df['pack_size']

# Extract whether the mask is for kids
products_df['is_kids'] = products_df['product_name'].str.lower().str.contains('kid|children|child')

# Extract language information from reviews
reviews_df['review_language'] = reviews_df['languageCode'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else 'unknown')

# Simplified text cleaning function without NLTK tokenization
def clean_text(text):
    if isinstance(text, str) and text.strip():
        # Convert to lowercase
        text = text.lower()
        # Remove special characters, numbers, etc.
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove stopwords (manual approach)
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    return ''

# Apply text cleaning to review text
reviews_df['clean_review_text'] = reviews_df['reviewText'].apply(clean_text)

# For non-English reviews, use the translated text if available
mask = (reviews_df['review_language'] != 'en') & reviews_df['translation.reviewText'].notna()
reviews_df.loc[mask, 'clean_review_text'] = reviews_df.loc[mask, 'translation.reviewText'].apply(clean_text)

# Perform sentiment analysis
print("Performing sentiment analysis...")
sia = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def get_sentiment(text):
    if isinstance(text, str) and text.strip():
        return sia.polarity_scores(text)['compound']
    return 0

reviews_df['sentiment_score'] = reviews_df['clean_review_text'].apply(get_sentiment)

# Start Analysis
print("\nAnalyzing product features and pricing...")

# 1. Distribution of mask types
plt.figure(figsize=(12, 6))
mask_type_counts = products_df['mask_type'].value_counts()
sns.barplot(x=mask_type_counts.index, y=mask_type_counts.values)
plt.title('Distribution of Face Mask Types', fontsize=16)
plt.ylabel('Count', fontsize=14)
plt.xlabel('Mask Type', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/mask_type_distribution.png')
print("Generated mask type distribution plot")

# 2. Analyze price distribution by mask type
plt.figure(figsize=(12, 6))
sns.boxplot(x='mask_type', y='price_per_mask', data=products_df)
plt.title('Price per Mask by Type', fontsize=16)
plt.ylabel('Price per Mask (AUD)', fontsize=14)
plt.xlabel('Mask Type', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/price_by_mask_type.png')
print("Generated price by mask type plot")

# 3. Calculate average ratings per product
product_ratings = reviews_df.groupby('productId')['rating_stars'].agg(['mean', 'count']).reset_index()
product_ratings.columns = ['product_id', 'avg_rating', 'review_count']

# Merge with product information
product_analysis = pd.merge(products_df, product_ratings, left_on='product_id', right_on='product_id', how='left')

# Display the top-rated products
top_rated = product_analysis.sort_values(by=['avg_rating', 'review_count'], ascending=False)
top_rated = top_rated[top_rated['review_count'] > 5]  # Filter for products with more than 5 reviews
print("\nTop-rated products (with >5 reviews):")
top_products = top_rated[['product_id', 'product_name', 'product_price', 'price_currency', 
                          'mask_type', 'avg_rating', 'review_count']].head(10)
print(top_products)

# Save to CSV for further analysis
top_products.to_csv('plots/top_rated_products.csv', index=False)

# 4. Analyze ratings distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='rating_stars', data=reviews_df)
plt.title('Distribution of Customer Ratings', fontsize=16)
plt.xlabel('Rating (Stars)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig('plots/ratings_distribution.png')
print("Generated ratings distribution plot")

# Group sentiment by mask type
# First, we need to merge reviews with products
reviews_with_product = pd.merge(reviews_df, products_df[['product_id', 'mask_type']], 
                               left_on='productId', right_on='product_id', how='left')

# 5. Analyze sentiment by mask type
sentiment_by_type = reviews_with_product.groupby('mask_type')['sentiment_score'].agg(['mean', 'count']).reset_index()
sentiment_by_type.columns = ['mask_type', 'avg_sentiment', 'review_count']

# Plot average sentiment by mask type
plt.figure(figsize=(12, 6))
# Use a bar plot with color intensity based on count
sns.barplot(x='mask_type', y='avg_sentiment', data=sentiment_by_type, palette='viridis')
plt.title('Average Sentiment by Mask Type', fontsize=16)
plt.xlabel('Mask Type', fontsize=14)
plt.ylabel('Average Sentiment Score (-1 to 1)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/sentiment_by_mask_type.png')
print("Generated sentiment by mask type plot")

print("\nCreating word clouds for positive and negative reviews...")
# 6. Create a word cloud of common terms in positive reviews
positive_reviews = reviews_df[reviews_df['sentiment_score'] > 0.5]['clean_review_text']
positive_text = ' '.join(positive_reviews.fillna(''))

# Generate a word cloud for positive reviews
plt.figure(figsize=(12, 8))
if positive_text.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis',
                        max_words=100).generate(positive_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Common Words in Positive Reviews', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/positive_reviews_wordcloud.png')
    print("Generated positive reviews wordcloud")

# 7. Create a word cloud of common terms in negative reviews
negative_reviews = reviews_df[reviews_df['sentiment_score'] < -0.3]['clean_review_text']
negative_text = ' '.join(negative_reviews.fillna(''))

# Generate a word cloud for negative reviews
plt.figure(figsize=(12, 8))
if negative_text.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='magma',
                        max_words=100).generate(negative_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Common Words in Negative Reviews', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/negative_reviews_wordcloud.png')
    print("Generated negative reviews wordcloud")

print("\nAnalyzing consumer segmentation...")
# 8. Analyze review patterns by language/region
language_stats = reviews_df.groupby('review_language').agg({
    'rating_stars': 'mean',
    'sentiment_score': 'mean',
    'productId': 'count'
}).reset_index()
language_stats.columns = ['language', 'avg_rating', 'avg_sentiment', 'review_count']
language_stats = language_stats.sort_values('review_count', ascending=False)

# Plot average rating by language
plt.figure(figsize=(14, 6))
top_languages = language_stats.head(10)
sns.barplot(x='language', y='avg_rating', data=top_languages)
plt.title('Average Rating by Language/Region', fontsize=16)
plt.ylabel('Average Rating (Stars)', fontsize=14)
plt.xlabel('Language', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/ratings_by_language.png')
print("Generated ratings by language plot")

print("\nIdentifying consumer segments based on review content...")
# 9. Identify different consumer segments based on their preferences
# First, extract key features from reviews
def has_keyword(text, keywords):
    if isinstance(text, str):
        text = text.lower()
        return any(keyword in text for keyword in keywords)
    return False

# Define different consumer segments based on review content
comfort_keywords = ['comfortable', 'soft', 'breathable', 'easy to breathe', 'breathe', 'comfort']
protection_keywords = ['protection', 'safe', 'secure', 'sealed', 'filter', 'filtering', 'safety']
style_keywords = ['stylish', 'design', 'color', 'look', 'fashion', 'cute', 'beautiful', 'pretty']
fit_keywords = ['fit', 'size', 'large', 'small', 'tight', 'loose', 'snug']
price_keywords = ['price', 'expensive', 'cheap', 'worth', 'value', 'cost']

# Create segment flags
reviews_df['comfort_focused'] = reviews_df['reviewText'].apply(lambda x: has_keyword(x, comfort_keywords))
reviews_df['protection_focused'] = reviews_df['reviewText'].apply(lambda x: has_keyword(x, protection_keywords))
reviews_df['style_focused'] = reviews_df['reviewText'].apply(lambda x: has_keyword(x, style_keywords))
reviews_df['fit_focused'] = reviews_df['reviewText'].apply(lambda x: has_keyword(x, fit_keywords))
reviews_df['price_focused'] = reviews_df['reviewText'].apply(lambda x: has_keyword(x, price_keywords))

# Also check translations for non-English reviews
columns_to_check = ['translation.reviewText']
for col in columns_to_check:
    if col in reviews_df.columns:
        for col_suffix, keywords in [
            ('comfort_focused', comfort_keywords),
            ('protection_focused', protection_keywords),
            ('style_focused', style_keywords),
            ('fit_focused', fit_keywords),
            ('price_focused', price_keywords)
        ]:
            mask = (~reviews_df[col_suffix]) & reviews_df[col].notna()
            reviews_df.loc[mask, col_suffix] = reviews_df.loc[mask, col].apply(
                lambda x: has_keyword(x, keywords)
            )

# Calculate the prevalence of each segment
segment_counts = {
    'Comfort Focused': reviews_df['comfort_focused'].sum(),
    'Protection Focused': reviews_df['protection_focused'].sum(),
    'Style Focused': reviews_df['style_focused'].sum(),
    'Fit Focused': reviews_df['fit_focused'].sum(),
    'Price Focused': reviews_df['price_focused'].sum()
}

# Plot segment distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=list(segment_counts.keys()), y=list(segment_counts.values()))
plt.title('Distribution of Consumer Segments', fontsize=16)
plt.ylabel('Count', fontsize=14)
plt.xlabel('Consumer Segment', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/consumer_segments.png')
print("Generated consumer segments distribution plot")

# 10. Analyze ratings by segment
segment_ratings = {
    'Comfort Focused': reviews_df[reviews_df['comfort_focused']]['rating_stars'].mean(),
    'Protection Focused': reviews_df[reviews_df['protection_focused']]['rating_stars'].mean(),
    'Style Focused': reviews_df[reviews_df['style_focused']]['rating_stars'].mean(),
    'Fit Focused': reviews_df[reviews_df['fit_focused']]['rating_stars'].mean(),
    'Price Focused': reviews_df[reviews_df['price_focused']]['rating_stars'].mean()
}

# Plot average ratings by segment
plt.figure(figsize=(12, 6))
sns.barplot(x=list(segment_ratings.keys()), y=list(segment_ratings.values()))
plt.title('Average Ratings by Consumer Segment', fontsize=16)
plt.ylabel('Average Rating (Stars)', fontsize=14)
plt.xlabel('Consumer Segment', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/ratings_by_segment.png')
print("Generated ratings by segment plot")

# 11. Analyze most common positive and negative aspects mentioned
def extract_common_aspects(reviews_text, n=20):
    all_words = ' '.join(reviews_text.fillna(''))
    words = all_words.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
    # Count frequencies
    word_freq = Counter(words)
    return pd.DataFrame(word_freq.most_common(n), columns=['Word', 'Frequency'])

# Extract most mentioned aspects in positive reviews
positive_aspects = extract_common_aspects(reviews_df[reviews_df['sentiment_score'] > 0.5]['clean_review_text'])
negative_aspects = extract_common_aspects(reviews_df[reviews_df['sentiment_score'] < -0.3]['clean_review_text'])

# Plot most common positive aspects
plt.figure(figsize=(12, 6))
if not positive_aspects.empty:
    sns.barplot(x='Frequency', y='Word', data=positive_aspects.head(15))
    plt.title('Most Common Words in Positive Reviews', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/common_positive_aspects.png')
    print("Generated common positive aspects plot")

# Plot most common negative aspects
plt.figure(figsize=(12, 6))
if not negative_aspects.empty:
    sns.barplot(x='Frequency', y='Word', data=negative_aspects.head(15))
    plt.title('Most Common Words in Negative Reviews', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/common_negative_aspects.png')
    print("Generated common negative aspects plot")

# 12. Analyze sentiment over time
reviews_df['review_month'] = pd.to_datetime(reviews_df['postedDate']).dt.strftime('%Y-%m')
time_sentiment = reviews_df.groupby('review_month')['sentiment_score'].mean().reset_index()

plt.figure(figsize=(14, 6))
if not time_sentiment.empty:
    sns.lineplot(data=time_sentiment, x='review_month', y='sentiment_score')
    plt.title('Average Sentiment Over Time', fontsize=16)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('plots/sentiment_over_time.png')
    print("Generated sentiment over time plot")

# 13. Find correlations between features and ratings
# Prepare a dataframe with all consumer segments
segment_df = reviews_df[['productId', 'comfort_focused', 'protection_focused', 'style_focused', 
                        'fit_focused', 'price_focused', 'rating_stars']]

# Group by product and calculate segment prevalence
product_segments = segment_df.groupby('productId').agg({
    'comfort_focused': 'mean',
    'protection_focused': 'mean',
    'style_focused': 'mean',
    'fit_focused': 'mean',
    'price_focused': 'mean',
    'rating_stars': 'mean'
}).reset_index()

# Calculate correlations
corr = product_segments.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation between Consumer Segments and Ratings', fontsize=16)
plt.tight_layout()
plt.savefig('plots/segment_rating_correlation.png')
print("Generated segment-rating correlation heatmap")

print("\nAnalysis complete! All visualizations have been saved to the 'plots' folder.")