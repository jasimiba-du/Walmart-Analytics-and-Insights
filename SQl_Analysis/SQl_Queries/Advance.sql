-- 1. Sales Contribution by Category in Each City
CREATE TEMP TABLE CityCategorySalesResults AS 
WITH CityCategorySales AS (
    SELECT 
        city, 
        category, 
        SUM(total_price) AS category_sales
    FROM 
        sales_data
    GROUP BY 
        city, category
), TotalCitySales AS (
    SELECT 
        city, 
        SUM(total_price) AS total_city_sales
    FROM 
        sales_data
    GROUP BY 
        city
)
SELECT 
    c.city, 
    c.category, 
    c.category_sales, 
    t.total_city_sales, 
    ROUND((c.category_sales / t.total_city_sales)::numeric, 2) AS sales_contribution_percentage
FROM 
    CityCategorySales c
JOIN 
    TotalCitySales t
ON 
    c.city = t.city
ORDER BY 
    c.city, sales_contribution_percentage DESC;

COPY CityCategorySalesResults TO 'E:/Projects/walmart_analytics/SQl_Analysis/Advance/sales_contribution_by_category.csv' WITH CSV HEADER;

-- 2. Peak Sales Hours
CREATE TEMP TABLE PeakSalesHoursResults AS
WITH HourlySales AS (
    SELECT 
        EXTRACT(HOUR FROM time) AS sales_hour, 
        SUM(total_price) AS hourly_sales
    FROM 
        sales_data
    GROUP BY 
        EXTRACT(HOUR FROM time)
)
SELECT 
    sales_hour, 
    hourly_sales, 
    RANK() OVER (ORDER BY hourly_sales DESC) AS sales_rank
FROM 
    HourlySales
ORDER BY 
    sales_rank;

COPY PeakSalesHoursResults TO 'E:/Projects/walmart_analytics/SQl_Analysis/Advance/peak_sales_hours.csv' WITH CSV HEADER;

-- 3. Top Performing Branches Based on Profit Margin
CREATE TEMP TABLE BranchProfitResults AS
WITH BranchProfit AS (
    SELECT 
        branch, 
        SUM(total_price * profit_margin) AS total_profit,
        SUM(total_price) AS total_revenue,
        ROUND((SUM(total_price * profit_margin) / SUM(total_price))::numeric, 2) AS profit_margin_percentage
    FROM 
        sales_data
    GROUP BY 
        branch
)
SELECT 
    branch, 
    total_profit, 
    total_revenue, 
    profit_margin_percentage, 
    RANK() OVER (ORDER BY profit_margin_percentage DESC) AS rank
FROM 
    BranchProfit
ORDER BY 
    rank;

COPY BranchProfitResults TO 'E:/Projects/walmart_analytics/SQl_Analysis/Advance/top_performing_branches.csv' WITH CSV HEADER;

-- 4. Customer Satisfaction by Category and City
CREATE TEMP TABLE CustomerSatisfactionResults AS
WITH CategoryRatings AS (
    SELECT 
        city, 
        category, 
        AVG(rating) AS avg_rating
    FROM 
        sales_data
    GROUP BY 
        city, category
), RankedCategories AS (
    SELECT 
        city, 
        category, 
        avg_rating, 
        RANK() OVER (PARTITION BY city ORDER BY avg_rating DESC) AS rank
    FROM 
        CategoryRatings
)
SELECT 
    city, 
    category, 
    avg_rating, 
    rank
FROM 
    RankedCategories
WHERE 
    rank <= 3
ORDER BY 
    city, rank;

COPY CustomerSatisfactionResults TO 'E:/Projects/walmart_analytics/SQl_Analysis/Advance/customer_satisfaction_by_category.csv' WITH CSV HEADER;

-- 5. Sales Trends Over Time
CREATE TEMP TABLE SalesTrendsResults AS
WITH MonthlySales AS (
    SELECT 
        TO_CHAR(date, 'YYYY-MM') AS sales_month, 
        SUM(total_price) AS monthly_sales
    FROM 
        sales_data
    GROUP BY 
        TO_CHAR(date, 'YYYY-MM')
), SalesGrowth AS (
    SELECT 
        sales_month, 
        monthly_sales, 
        LAG(monthly_sales) OVER (ORDER BY sales_month) AS previous_month_sales,
        ROUND(((monthly_sales - LAG(monthly_sales) OVER (ORDER BY sales_month)) / 
               LAG(monthly_sales) OVER (ORDER BY sales_month))::numeric, 2) AS growth_percentage
    FROM 
        MonthlySales
)
SELECT 
    sales_month, 
    monthly_sales, 
    previous_month_sales, 
    growth_percentage
FROM 
    SalesGrowth
ORDER BY 
    sales_month;

COPY SalesTrendsResults TO 'E:/Projects/walmart_analytics/SQl_Analysis/Advance/sales_trends_over_time.csv' WITH CSV HEADER;

-- 6. Payment Method Preference by City
CREATE TEMP TABLE PaymentMethodResults AS
WITH PaymentPreference AS (
    SELECT 
        city, 
        payment_method, 
        COUNT(*) AS usage_count
    FROM 
        sales_data
    GROUP BY 
        city, payment_method
), RankedPaymentMethods AS (
    SELECT 
        city, 
        payment_method, 
        usage_count, 
        RANK() OVER (PARTITION BY city ORDER BY usage_count DESC) AS rank
    FROM 
        PaymentPreference
)
SELECT 
    city, 
    payment_method, 
    usage_count
FROM 
    RankedPaymentMethods
WHERE 
    rank = 1
ORDER BY 
    city;

COPY PaymentMethodResults TO 'E:/Projects/walmart_analytics/SQl_Analysis/Advance/payment_method_preference_by_city.csv' WITH CSV HEADER;
