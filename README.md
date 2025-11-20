# Loan Interest Rates Analysis with Python

## Overview
I've joined a fintech startup tasked with improving customer loan offers. Using the [Bondora P2P Loans](https://www.kaggle.com/datasets/marcobeyer/bondora-p2p-loans?select=LoanData.csv) dataset, I'll build build insights about what factors help determine a person's interest rate. I will work your way to creating a predictive model to estimate loan interest rates, which will guide your company in personalizing loan terms efficiently.

The features used in this project are:

VerificationType: Method used for loan application data verification
Age: Age of the borrower (years)
AppliedAmount: Amount applied
Amount: Amount the borrower received
Interest: Interest rate
LoanDuration: The loan term
Education: Education of the borrower
EmploymentDurationCurrentEmployer: Employment time with the current employer
HomeOwnershipType: Home ownership type
IncomeTotal: Total income
ExistingLiabilities: Borrower's number of existing liabilities
RefinanceLiabilities: The total amount of liabilities after refinancing
Rating: Bondora Rating issued by the Rating model
NoOfPreviousLoansBeforeLoan: Number of previous loans
AmountOfPreviousLoansBeforeLoan: Value of previous loans

## Task 1: Importing libraries
Before I start working on the dataset, it is good practice to import all libraries at the beginning of my code.


## Task 2:  Load and Clean the Data

### Load the Data

Now that I have imported the right libraries, I can use Pandas to load the data from the CSV.



### Cleaning the Dataset

Before starting my analysis, take some time to familiarize myself with the dataset to understand the available information. This is also a good opportunity to clean the data by removing missing values and adjusting the index.

#### Exercise 1: Set the Index
Explore the dataset to find the number of rows, columns, and data types of each feature. Identify any missing values.

## Task 3: Data Analysis

<img width="926" height="517" alt="Screenshot (11)" src="https://github.com/user-attachments/assets/413f6dc7-1d72-415a-947f-603606afdb5c" />

<img width="944" height="477" alt="Screenshot (12)" src="https://github.com/user-attachments/assets/1cf0b325-57c1-43c0-879b-a9feeb0a9db6" />

<img width="410" height="256" alt="Screenshot (13)" src="https://github.com/user-attachments/assets/10e0a307-66f1-4a69-8329-36f3329e913c" />

<img width="1183" height="398" alt="Screenshot (14)" src="https://github.com/user-attachments/assets/0ebd594a-62eb-46bb-9d5b-6ac239fee8ac" />

<img width="1177" height="400" alt="Screenshot (15)" src="https://github.com/user-attachments/assets/25934891-1310-40ad-a487-c942b5f2232b" />

<img width="744" height="465" alt="Screenshot (16)" src="https://github.com/user-attachments/assets/4487feac-4472-44f4-ac76-8d6c7542f4a5" />

<img width="392" height="280" alt="Screenshot (17)" src="https://github.com/user-attachments/assets/bc94e6c0-c770-4547-b447-bda7579cf637" />

<img width="555" height="506" alt="Screenshot (18)" src="https://github.com/user-attachments/assets/072ea586-e744-4b35-ad2e-63891046f3d7" />

<img width="615" height="356" alt="Screenshot (19)" src="https://github.com/user-attachments/assets/81ce4ea6-1124-4c5a-8bf1-5272d917247f" />

## Task 4: Sales Trend over Time

<img width="754" height="329" alt="Screenshot (20)" src="https://github.com/user-attachments/assets/06b6e02f-1ea3-41e2-bda1-fc569666d954" />

<img width="864" height="509" alt="Screenshot (21)" src="https://github.com/user-attachments/assets/e0b033a0-6f32-4100-a8cd-381cd22b757d" />

<img width="737" height="524" alt="Screenshot (22)" src="https://github.com/user-attachments/assets/c2b76e41-c059-47c5-81e7-d747a8682e32" />

<img width="650" height="533" alt="Screenshot (23)" src="https://github.com/user-attachments/assets/8736ecca-9a9c-42bb-9912-838997031285" />

<img width="1175" height="533" alt="Screenshot (24)" src="https://github.com/user-attachments/assets/75827b52-f98e-4f2c-9fc7-af16b2bcf394" />

<img width="303" height="158" alt="Screenshot (25)" src="https://github.com/user-attachments/assets/fb01f6c6-b4e7-4afa-8411-356607e3e558" />

<img width="1164" height="508" alt="Screenshot (26)" src="https://github.com/user-attachments/assets/1ba82c6f-f25f-4812-a712-c5ff3bf1f59a" />

<img width="1176" height="486" alt="Screenshot (27)" src="https://github.com/user-attachments/assets/cdcd55db-a5c1-4736-bbea-f00d3d9be8ed" />

<img width="290" height="126" alt="Screenshot (28)" src="https://github.com/user-attachments/assets/4d1d8228-26c0-4444-b76b-3264441e65dc" />

<img width="1166" height="507" alt="Screenshot (29)" src="https://github.com/user-attachments/assets/4147bc01-2493-4df7-9f9f-dd0ef319c48b" />

<img width="387" height="82" alt="Screenshot (30)" src="https://github.com/user-attachments/assets/8d3bd363-5796-4409-84f7-6f27e672b811" />

## Task 5: Top-selling Products and Countries

<img width="634" height="509" alt="Screenshot (31)" src="https://github.com/user-attachments/assets/dd7179d2-52ad-4992-8027-452ade9e9ff5" />

<img width="868" height="520" alt="Screenshot (32)" src="https://github.com/user-attachments/assets/e190204c-927f-404a-b7a4-c2042014dba3" />

<img width="1174" height="516" alt="Screenshot (33)" src="https://github.com/user-attachments/assets/a9766b1d-45f5-48ab-9689-cff449b4b371" />

<img width="485" height="101" alt="Screenshot (34)" src="https://github.com/user-attachments/assets/b46e5868-18b7-4e02-b7d7-20ac1b4d4094" />

<img width="1174" height="366" alt="Screenshot (35)" src="https://github.com/user-attachments/assets/437e97af-f938-4361-8aaa-d07992053360" />

<img width="1170" height="377" alt="Screenshot (36)" src="https://github.com/user-attachments/assets/081a74e7-ebc3-46bc-be18-a08e906db411" />

<img width="545" height="232" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/0e6ab90a-a68b-4c6d-b1b4-9d4314fed67b" />

## Task 6: Outlier Detection and Analysis

<img width="790" height="520" alt="Screenshot (38)" src="https://github.com/user-attachments/assets/f247c6df-3f46-469a-bd35-de730e98545c" />

<img width="352" height="232" alt="Screenshot (40)" src="https://github.com/user-attachments/assets/debabf7b-06c5-418f-ba9b-51aec275cd37" />

<img width="1190" height="390" alt="Screenshot (41)" src="https://github.com/user-attachments/assets/cc6fbc12-0b25-4db3-b571-2390a4667d40" />

<img width="1132" height="392" alt="Screenshot (42)" src="https://github.com/user-attachments/assets/a0712aea-ccef-461a-ab09-a1a16caa5ee4" />

<img width="697" height="422" alt="Screenshot (43)" src="https://github.com/user-attachments/assets/7b3a8d97-cb12-4f6e-929a-dad3b0746762" />

## Task 7: Customer Purchase Pattern and Summary

<img width="720" height="528" alt="Screenshot (44)" src="https://github.com/user-attachments/assets/73adacd8-d9bf-4121-b559-d2d06575a5c7" />

<img width="1183" height="494" alt="Screenshot (45)" src="https://github.com/user-attachments/assets/29779261-5706-4c3b-a6f2-a9e719d7843f" />

<img width="409" height="429" alt="Screenshot (46)" src="https://github.com/user-attachments/assets/b2641809-3f68-41bd-ac17-c901a4934897" />

<img width="320" height="172" alt="Screenshot (47)" src="https://github.com/user-attachments/assets/902096e3-4671-4d34-bac4-394c04c30810" />

## Task 8: Key findings and Conclusion

<img width="1161" height="514" alt="Screenshot (48)" src="https://github.com/user-attachments/assets/186de33c-2fc0-4554-a058-7d36959b832e" />

<img width="756" height="524" alt="Screenshot (49)" src="https://github.com/user-attachments/assets/50a4d171-d934-4c99-9f95-ec8c3948a1db" />

<img width="832" height="521" alt="Screenshot (50)" src="https://github.com/user-attachments/assets/4cd98f35-9ddc-495f-a918-5534de4413da" />

<img width="671" height="497" alt="Screenshot (51)" src="https://github.com/user-attachments/assets/c964e35c-0773-4652-953f-80114bf54f7e" />

<img width="928" height="514" alt="Screenshot (52)" src="https://github.com/user-attachments/assets/cb2e1a6d-09c4-45ab-9da7-1a889d0de39b" />

<img width="1183" height="520" alt="Screenshot (53)" src="https://github.com/user-attachments/assets/a4944961-85c3-460e-8de8-a282cb4aed0f" />

<img width="1094" height="527" alt="Screenshot (54)" src="https://github.com/user-attachments/assets/0a280617-fe7f-4875-80ab-6877dd006b72" />

## Executive Summary

<img width="814" height="535" alt="Screenshot (55)" src="https://github.com/user-attachments/assets/19b05dcc-ea94-41c7-bf77-b165fa1ce165" />

Based on the comprehensive analysis of the online retail dataset, here are the most important insights:

1. **Sales Patterns**:
   - Sales show clear seasonal patterns with specific peak months
   - Weekdays generate significantly more revenue than weekends
   - Specific hours of the day show consistently higher transaction volumes

2. **Geographic Distribution**:
   - Sales are heavily concentrated in a few key markets
   - The UK likely represents the primary market with the highest sales volume
   - International markets show varying levels of engagement and potential for growth

3. **Product Performance**:
   - Gift items and home accessories appear frequently among top sellers
   - There is a notable difference between products that sell in high quantities versus those that generate high revenue
   - Certain products show seasonal popularity patterns

4. **Customer Behavior**:
   - The customer base follows the Pareto principle (roughly 20% of customers generate 80% of revenue)
   - Purchase frequency varies widely across the customer base
   - Average order values suggest a mid-range retail price point

5. **Outliers and Special Cases**:
   - Bulk orders significantly impact average metrics
   - Some extremely high-value transactions suggest B2B or wholesale activity
   - Unusual purchasing patterns may indicate special events or promotional activities

These insights can guide strategic decisions in inventory management, marketing campaigns, customer retention programs, and international expansion efforts. The data suggests opportunities for targeted promotions during slower periods, loyalty programs for high-value customers, and specialized approaches for different market segments.

