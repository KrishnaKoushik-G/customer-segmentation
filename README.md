# Customer Segmentation and Product Recommendation

## Abstract
In the online retail landscape, understanding customer behavior and preferences is crucial for effective marketing and sales. Segmenting customers based on transactional and behavioral similarities provides valuable insights into spending patterns and product preferences. This project aims to transform raw transactional data into a customer-centric dataset suitable for segmentation analysis. It involves data cleaning to handle missing values, duplicates, and outliers, and feature engineering using the RFM (Recency, Frequency, Monetary) model. We employ the K-means clustering algorithm to partition customers into distinct segments. These segments are then evaluated, and a recommendation system is developed to enhance cross-selling opportunities. The system suggests top-selling items to customers within each segment, increasing marketing effectiveness, sales, and customer loyalty. By integrating advanced analytics and machine learning, this project enhances customer experiences and promotes sustainable business growth. 

## 1. Introduction

In online retail, understanding customer behavior and preferences is crucial for effective marketing and sales growth. Customer segmentation, which groups customers based on shared characteristics and behaviors, allows for tailored offerings and communication strategies. This enhances customer engagement and loyalty by leveraging demographic information, purchasing patterns, and other metrics.

### 1.1 Customer Behavior and Features

Customer lifetime value (CLV) is essential in segmentation, reflecting the potential long-term revenue from a customer. By focusing on CLV, marketers can optimize resources and strategies to foster long-term relationships and sustained growth. The RFM (Recency, Frequency, Monetary) model is also vital, helping categorize customers based on how recently they made a purchase, how often they buy, and how much they spend. This detailed understanding allows for targeted and personalized marketing efforts.

### 1.2 Customer Segmentation

Various methodologies, including traditional demographic segmentation and advanced machine learning techniques, are used for customer segmentation. Hierarchical clustering and K-means clustering are prominent examples. K-means clustering, known for its scalability and efficiency, partitions customers based on purchasing behavior and preferences, providing actionable insights for targeted marketing and personalized recommendations.

### 1.3 Product Recommendation System

Product recommendation systems enhance the shopping experience by delivering personalized suggestions based on customer data. By integrating segmentation analyses, these systems can tailor recommendations to specific customer groups, increasing cross-selling opportunities and sales. Continuous refinement through machine learning ensures these systems adapt to changing customer preferences and market trends.

In this project, we aim to leverage customer segmentation techniques, particularly K-means clustering. By clustering customers into distinct groups based on their purchasing behaviour, we intend to identify common preferences and characteristics within each segment. Subsequently, we will develop a product recommendation system that suggests top-selling products to customers within each segment who have not yet made corresponding purchases. Through this approach, we aim to enhance marketing effectiveness, drive sales growth, and foster increased customer satisfaction in the competitive landscape of online retail. 

## 2. Methodology and Discussion

### 2.1	Dataset Description 

The dataset is sourced from the UCI Machine Learning Repository, documenting all transactions from a UK-based retailer transpired between the years 2010 and 2011. It dataset comprises 541,909 entries meticulously curated across 8 columns.
| Variable    | Description                                                                                 |
|-------------|---------------------------------------------------------------------------------------------|
| `InvoiceNo` | A 6-digit code representing each unique transaction. If this code starts with the letter 'c', it indicates a cancellation. |
| `StockCode` | A 5-character code uniquely assigned to each distinct product.                              |
| `Description`| Name/Description of each product.                                                          |
| `Quantity`  | The number of units of a product in a transaction.                                          |
| `InvoiceDate`| The date and time of the transaction.                                                      |
| `UnitPrice` | The unit price of the product in sterling.                                                  |
| `CustomerID`| Identifier uniquely assigned to each customer.                                              |
| `Country`   | The country of the customer.                                                                |

### 2.2 Libraries Imported
To facilitate data manipulation, analysis, and modeling, we imported several essential libraries:

- **NumPy**: For numerical computations and array operations.
- **Pandas**: For efficient data handling using data frames and series.
- **Seaborn and Matplotlib**: For creating informative and visually appealing graphs and charts. Seaborn provides a higher-level interface for statistical visualization.
- **Scikit-learn**: For accessing the KMeans clustering algorithm and evaluating clustering performance using silhouette scores.

These libraries are crucial for data preprocessing, clustering, and evaluation, setting the foundation for developing an effective product recommendation system tailored to segmented customer groups.

### 2.3 Exploratory Data Analysis

Exploratory Data Analysis (EDA) is essential for gaining insights, identifying patterns, and formulating hypotheses before advanced modeling. It includes examining and visualizing data to understand its structure, distribution, and relationships.

#### 2.3.1 Summary Statistics

Summary statistics provide an overview of the dataset's central tendency, dispersion, and shape. Key measures include mean, median, standard deviation, quantiles, and percentiles. These statistics help identify potential outliers or anomalies.

- **Average Quantity**: Approximately 9.55 per transaction, with a wide range indicating returned or canceled orders.
- **Average Unit Price**: Around 4.61, with anomalies like negative prices suggesting data errors.
- **CustomerID**: 406,829 non-null entries, indicating missing values.
- **InvoiceNo**: 25,900 unique transactions, with some indicating bulk orders.
- **Stock Codes and Product Descriptions**: 4,070 unique stock codes and 4,223 unique descriptions, with frequent items indicating popular products.
- **Country Distribution**: Transactions from 38 countries, with the UK accounting for about 91.4% of the total.

#### 2.3.2 Visualization

Visualization techniques, such as histograms, box plots, scatter plots, and heatmaps, are used to explore data. These tools help identify patterns and dependencies within the dataset.

- **Figure 4**: Frequency distribution of the Quantity column.
- **Figure 5**: Country-wise sales distribution.

#### 2.3.3 Initial Data Cleaning and Preprocessing

Data cleaning and preprocessing ensure dataset quality and integrity. This includes handling missing values, outliers, and inconsistencies, and transforming variables to meet modeling assumptions.

- **Missing Values**: The dataset contains 1,456 null values in the Description column and 135,080 null values in the CustomerID column. Rows with missing CustomerIDs are removed to maintain cluster integrity. Similarly, rows with missing Descriptions are removed to prevent errors.
- **Duplicates**: 5,225 duplicate rows are removed to avoid noise and inaccuracies in clustering.
- **Zero Unit Prices**: Transactions with zero unit prices (33 occurrences) are removed to maintain data consistency.

After cleaning, the dataset is reduced to 401,604 rows, ensuring more accurate and insightful analyses.

### 2.4 Feature Engineering

In developing a robust customer-centric dataset for clustering and recommendations, we undertake feature engineering to extract insights and enhance analytical capabilities.

#### RFM Analysis

**Recency (R):**
- **Days Since Last Purchase:** Measures the days since the last purchase, indicating customer engagement.

**Frequency (F):**
- **Total Transactions:** Counts the number of transactions per customer.
- **Total Products Purchased:** Sums the quantity of products bought across all transactions.

**Monetary (M):**
- **Total Spend:** Total expenditure by a customer (sum of UnitPrice * Quantity).
- **Average Transaction Value:** Total Spend divided by Total Transactions.

#### Additional Features

**Product Diversity:**
- **Unique Products Purchased:** Number of distinct products a customer has bought, indicating diverse tastes.

**Behavioral Insights:**
- **Average Days Between Purchases:** Average interval between consecutive purchases, helping predict future buying patterns.

**Cancellation Insights:**
- **Cancellation Frequency:** Number of cancelled transactions per customer.
- **Cancellation Rate:** Proportion of cancelled transactions to total transactions.
  
These features offer a detailed view of customer behavior and preferences, enabling precise customer segmentation and personalized marketing strategies.

| Variable                       | Description                                                                                             |
|--------------------------------|---------------------------------------------------------------------------------------------------------|
| `CustomerID`                   | Identifier uniquely assigned to each customer, used to distinguish individual customers.                |
| `Days_Since_Last_Purchase`     | The number of days that have passed since the customer's last purchase.                                 |
| `Total_Transactions`           | The total number of transactions made by the customer.                                                  |
| `Total_Products_Purchased`     | The total quantity of products purchased by the customer across all transactions.                       |
| `Total_Spend`                  | The total amount of money the customer has spent across all transactions.                               |
| `Average_Transaction_Value`    | The average value of the customer's transactions, calculated as total spend divided by the number of transactions. |
| `Unique_Products_Purchased`    | The number of different products the customer has purchased.                                            |
| `Average_Days_Between_Purchases`| The average number of days between consecutive purchases made by the customer.                         |
| `Cancellation_Frequency`       | The total number of transactions that the customer has cancelled.                                       |
| `Cancellation_Rate`            | The proportion of transactions that the customer has cancelled, calculated as cancellation frequency divided by total transactions. |

Table 2: Description of columns of the customer_data 

### 2.5 Data Preprocessing

Data preprocessing is essential for preparing raw data, ensuring its quality and usability. This phase involves handling missing values, removing duplicates, and transforming data into an analysis-ready format, laying the groundwork for accurate insights.

#### 2.5.1 Initial Steps

To refine the 'customer_data' dataframe, we checked for columns with identical values and found none. All columns contained distinct and informative data, including the crucial CustomerID.

#### 2.5.2 Outlier Treatment

Outliers, data points that deviate significantly from the dataset, can affect statistical analysis and models. We identify outliers using summary statistics and visualizations like box plots.

**Strategies for Outlier Treatment:**
- **Removal:** Eliminating data points likely due to errors.
- **Transformation:** Adjusting extreme values using methods like logarithmic transformation.
- **Imputation:** Estimating missing or extreme values using mean, median, or mode.
- **Robust Methods:** Using statistical techniques that minimize outliers' influence.
- **Modeling Techniques:** Applying models inherently robust to outliers, like tree-based algorithms.

**IQR Method:**
We use the interquartile range (IQR) to identify outliers:
\[ \text{IQR} = Q3 - Q1 \]
\[ \text{Lower Bound} = Q1 - 1.5 \times \text{IQR} \]
\[ \text{Upper Bound} = Q3 + 1.5 \times \text{IQR} \]

Data points outside these bounds are considered outliers. We iteratively examine each numeric variable and adjust outliers accordingly. Outlier treatment ensures the dataset's reliability, stability, and interpretability, supporting accurate predictive models and informed decision-making.

