-- 1.1. Top-performing cities by revenue  
COPY (SELECT city, SUM(total_price) AS total_sales FROM sales_data GROUP BY city ORDER BY total_sales DESC LIMIT 10) TO 'E:/Projects/walmart_analytics/SQl_Analysis/Basic/output1.csv' WITH CSV HEADER;

-- Top ten most profit margin generated branches  
COPY (SELECT branch, AVG(profit_margin) AS avg_profit_margin FROM sales_data GROUP BY branch ORDER BY avg_profit_margin DESC LIMIT 10) TO 'E:/Projects/walmart_analytics/SQl_Analysis/Basic/output2.csv' WITH CSV HEADER;

-- Top categories by total sales in each city with year  
COPY (SELECT city, category, EXTRACT(YEAR FROM date) AS year, SUM(total_price) AS total_sales FROM sales_data GROUP BY city, category, year ORDER BY city, total_sales DESC) TO 'E:/Projects/walmart_analytics/SQl_Analysis/Basic/output3.csv' WITH CSV HEADER;

-- 2.1. Customer ratings by payment method  
COPY (SELECT payment_method, AVG(rating) AS avg_rating FROM sales_data GROUP BY payment_method ORDER BY avg_rating DESC LIMIT 10) TO 'E:/Projects/walmart_analytics/SQl_Analysis/Basic/output4.csv' WITH CSV HEADER;

-- 2.2. Average profit margin by category  
COPY (SELECT category, AVG(profit_margin) AS avg_profit FROM sales_data GROUP BY category LIMIT 10) TO 'E:/Projects/walmart_analytics/SQl_Analysis/Basic/output5.csv' WITH CSV HEADER;

-- 3.1. Monthly sales trends across branches  
COPY (SELECT TO_CHAR(date, 'YYYY-MM') AS month, branch, SUM(total_price) AS total_sales FROM sales_data GROUP BY TO_CHAR(date, 'YYYY-MM'), branch ORDER BY month ASC) TO 'E:/Projects/walmart_analytics/SQl_Analysis/Basic/output6.csv' WITH CSV HEADER;

-- 4.1. Ewallet satisfaction comparison (ratings)  
COPY (SELECT payment_method, rating FROM sales_data WHERE payment_method = 'Ewallet' OR payment_method = 'Cash' LIMIT 10) TO 'E:/Projects/walmart_analytics/SQl_Analysis/Basic/output7.csv' WITH CSV HEADER;
