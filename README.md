# Review classification and sentiment analysis

 ## Table of Contents
1. Introduction
2. Dataset
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Data Preprocessing
6. Modeling and Results
7. Web App screenshots
   

### 1. Introduction ###

The Clothing Review Sentiment Analysis project is aimed at predicting the sentiment of customer reviews for various clothing items. The goal is to classify reviews as positive or negative based on the content of the review text.

The project leverages Natural Language Processing (NLP) techniques and machine learning models to achieve high accuracy in sentiment classification. This project is valuable for retailers who wish to gain insights into customer satisfaction and improve their products based on feedback.

## 2. Dataset ## 

[Clothing Reviews-Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
It includes the following columns:

- Clothing ID: Unique identifier for each clothing item.
- Age: Age of the reviewer.
- Title: Title of the review.
- Review Text: The actual text of the review.
- Rating: Rating given by the reviewer (1-5 scale).
- Recommended IND: Indicator of whether the reviewer recommends the item.
- Positive Feedback Count: Number of positive feedbacks received by the review.
- Division Name, Department Name, Class Name: Metadata about the clothing item.
  
## 3. Data Cleaning ##
Prior to conducting exploratory data analysis (EDA), a comprehensive data cleaning process was undertaken to ensure the dataset's quality and integrity. The following steps were applied:

- Handling Missing Values: All instances of missing data were systematically addressed. Missing entries in critical columns were either imputed with suitable values or removed, depending on the context and impact on downstream analysis.

- Class Imbalance Management: The dataset exhibited a significant imbalance between the two sentiment classes. To mitigate this, `undersampling was employed on the majority class`, bringing the dataset to a more balanced state. This step was crucial in ensuring that the models trained on the data were not biased towards the more prevalent class, thereby improving the robustness of the sentiment classification.
 - ![image](https://github.com/user-attachments/assets/be523c7e-157e-4052-8ee7-ddc206ee4f93) 


- Removal of Anomalous Entries: Certain entries were identified as biased or inconsistent, such as `instances where a rating of 5 was given, yet the recommendation indicator was 0`. These entries were removed to prevent any distortions in the model's learning process, ensuring that the training data accurately reflected the true sentiment of the reviewers.
  
## 4. Exploratory Data Analysis (EDA) ##

Before diving into model building, an extensive exploratory data analysis was conducted utilizing **`plotly, seaborn and matplotlib`**. This included:

- **Distribution of Ratings** : Visualization of how ratings and recommended class are distributed across the dataset. Here are some of the visualizations:
  - ![image](https://github.com/user-attachments/assets/f836db7b-3ac6-4b31-9059-d64d7b8d4c3c)

  - ![image](https://github.com/user-attachments/assets/f7c23002-a38e-461d-be03-fc0e872c12be)

- **Word Clouds**: Created word clouds to visualize the most frequent words in positive and negative reviews.
  - Positive reviews
  ![image](https://github.com/user-attachments/assets/88f20fdd-e80f-4cb7-a76c-b12ecae2ce5d)
  - Negative reviews
    ![image](https://github.com/user-attachments/assets/7a073076-48c6-4ee5-a7f5-d2892d85deb2)

- **Correlation Analysis**: Checked for correlations between different features and review sentiments.

#  5. Data Preprocessing #

To prepare the data for modeling, the following preprocessing steps were undertaken:

- Text Cleaning: Removed HTML tags, special characters, and numbers from the review text.
- Stopword Removal: Common stopwords were removed to reduce noise in the data.
- Lemmatization: Converted words to their base form using lemmatization to standardize the text.
- TF-IDF Vectorization: Transformed the text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).

## 6. Modeling and Results ##
A variety of machine learning models were tested to classify the reviews, with Logistic regression, SVM and decision tree standing out across different metrics:

![image](https://github.com/user-attachments/assets/47068f4a-dccb-4318-b48b-44ab16ef1ab8)

## 7. Web app screenshots ##
## Negative review ##

![image](https://github.com/user-attachments/assets/6e48e830-7c08-4a2e-a12d-6db83b34fba5)

## Postive review ##

![image](https://github.com/user-attachments/assets/dc1f9bfd-02a9-495f-bb67-000b06942503)
