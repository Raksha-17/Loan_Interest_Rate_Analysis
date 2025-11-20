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

<img width="261" height="123" alt="Screenshot (73)" src="https://github.com/user-attachments/assets/69a84e7c-ad2f-468c-b38f-f998f5a05bfa" />

## Task 2:  Load and Clean the Data

### Load the Data
Now that I have imported the right libraries, I can use Pandas to load the data from the CSV.

<img width="963" height="536" alt="Screenshot (74)" src="https://github.com/user-attachments/assets/28ac6871-3ca9-4e79-82ad-4cd0f12a09ce" />

### Cleaning the Dataset

Before starting my analysis, take some time to familiarize myself with the dataset to understand the available information. This is also a good opportunity to clean the data by removing missing values and adjusting the index.

#### Exercise 1: Set the Index
Explore the dataset to find the number of rows, columns, and data types of each feature. Identify any missing values.

<img width="347" height="183" alt="Screenshot (75)" src="https://github.com/user-attachments/assets/eb552c0d-fa7a-40ba-9a7c-f7d350509d1d" />

## Task 3: Retrieving key metrics
To recommend loan offers, take a moment to understand the loan amounts and ratings. I’ll also want to get a rough idea of the interest rates being paid.

### Describing the dataset

#### Exercise 2: descriptive statistics

<img width="758" height="327" alt="Screenshot (77)" src="https://github.com/user-attachments/assets/3fb5ac3a-e19d-4ebc-9c9d-58d1e33f61ab" />

<img width="532" height="240" alt="Screenshot (78)" src="https://github.com/user-attachments/assets/a779764b-7b27-4136-ac83-91d0c26cc09e" />

### High-Risk Customers
Customers with a high debt-to-income ratio and less job stability may have more difficulty repaying loans, making them riskier.


#### Exercise 3: Identifying High-Risk Customers
I want to identify these customers and be able to add a flag to their loans. I consider borrowers to have less job stability if they have been on the current job for less than 1 year (including those in the trial period). In this scenario, a loan-to-rate ratio above 0.35 is considered risky.

<img width="896" height="449" alt="Screenshot (79)" src="https://github.com/user-attachments/assets/24ef4ce5-45f4-45d9-9c9b-ad389cac6a05" />

#### Implications:

* Risky loans mean interest rate: 28.86% Non-risky loans mean interest rate: 26.99%

* Risk premium: risky loans are about 1.87 percentage points higher than non-risky loans (28.86% − 26.99% ≈ 1.87%). Does this reflect a riskier nature?

* Yes. The higher rate for riskier loans is consistent with risk-based pricing: lenders charge more to compensate for the higher likelihood of default. Since risky loans only make up about 15.88% of the data, their impact on the overall average rate is present but modest. The total dataset rate should sit between the two subgroup means and be closer to the non-risky rate due to the larger share of non-risky loans.

## Task 4: Understanding different customer profiles

As a fintech analyst, understanding customer profiles allows me to identify patterns. I want to understand how different factors of the borrowers influence loan applications.

### Visualization of different profiles
To help my identify the different profiles, I decide to use my visualization skills to uncover actionable patterns for tailoring loan offers.

#### Exercise 4: Segmentation using box plots

<img width="728" height="356" alt="Screenshot (80)" src="https://github.com/user-attachments/assets/68354748-1a01-4113-9061-76e65a792155" />

<img width="939" height="538" alt="Screenshot (81)" src="https://github.com/user-attachments/assets/ce5c4a14-61ce-41c3-aa55-9982d6545466" />

#### Exercise 5: Scatter plots and correlation
I also want to investigate the relationship between certain numerical features and the interest rate. For that, I decide to use scatter plots, along with the correlation between features.

<img width="598" height="463" alt="Screenshot (82)" src="https://github.com/user-attachments/assets/45cc4c63-fbb5-419e-8205-3887d02176f1" />

<img width="945" height="485" alt="Screenshot (83)" src="https://github.com/user-attachments/assets/461e50c2-54a1-47b5-a0aa-5f2c6204770f" />

<img width="925" height="533" alt="Screenshot (85)" src="https://github.com/user-attachments/assets/794425bd-bebf-46c4-af75-4c36e9a59f4f" />

<img width="714" height="459" alt="Screenshot (86)" src="https://github.com/user-attachments/assets/8d9dee15-8ee6-49c0-81a3-75c4bd09b062" />

#### Implication:
* AmountOfPreviousLoansBeforeLoan (r = -0.175) This is the only feature among those four with a modest negative linear relationship to interest rate. In practical terms: Borrowers with more prior loans tend to receive slightly lower interest rates on new loans.Having more prior loans could signal reliability or familiarity with debt management, which modestly mitigates default risk, hence a cheaper rate.

* The negative sign aligns with risk-based pricing: more established credit behavior tends.

* The effect is small, but it’s the strongest signal among the four features you reported.


### Applied and Received Amounts
I noticed that there are two similar columns, "AppliedAmount" and "Amount", in the dataset. This implies that sometimes borrowers get loaned a different amount than what they asked for.

#### Exercise 6: Confidence Intervals
If more than 5% of loans are approved for less than requested, the team may need to revise how loan amounts are communicated to applicants. Estimate this proportion using a confidence interval to support your recommendation.

The proportion of loans where the requested amount differs from the given amount is pretty small, so it should be safe to only analyze one of those columns.

## Task 5: Modelling the Interest Rate
To make personalized loan offers, I decide to go one step further in my analysis and build a model to predict interest rates using different customer features. This will help me both be able to predict interest rates for new customers, and observe which features are actually statistically significant in determining the interest rates.


### Simple Linear Regression
To get my first model going I begin creating a simple linear regression. Based on the correlation analysis I did before, a good candidate for the independent variable is "AmountOfPreviousLoansBeforeLoan", which presented the strongest correlation with the target variable "Interest".

#### Exercise 7: Training the Linear Regression




Use this next cell to make some further analysis on my model. I are already given the line of best fit graph, but can add as many visualizations as I wish.




The predictor AmountOfPreviousLoansBeforeLoan has a statistically significant negative association with Interest (coefficient ≈ -0.0006, p < 0.001), but the model’s R-squared is only 0.031. So it’s unlikely this single predictor meaningfully explains variability in interest.

### Building a More Complex Linear Regression Model
Since predicting the interest rate using a single variable didn’t yield strong results, I decide to take a more comprehensive approach. This time, I’ll build a more complex model that includes multiple variables—possibly even some categorical ones.

This time I would use VerificationType, NoOfPreviousLoansBeforeLoan, AmountOfPreviousLoansBeforeLoan, Rating as my predictors.

#### Exercise 8: Building and refining the model


## Key Summary:

The model explains about 60% of the variation in Interest (R-squared ≈ 0.605). The strongest drivers are the rating categories (Rating_AA through Rating_HR) and verification indicators, with large, highly significant coefficients for mid-to-high ratings (D–HR). The intercept is stable at around 15.1, representing the baseline Interest at the reference levels. This setup achieves a good balance between explanatory power and interpretability: the ratings and verification status provide clear, actionable signals about pricing.
