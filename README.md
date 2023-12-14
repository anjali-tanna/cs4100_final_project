# CS4100 Final Project
## [Addressing Political Bias in News Articles with Multinomial Regression](https://docs.google.com/document/d/1Ikoay_N5WoR-IqjMdaVAHz-y5ivq-MUC452NJHw-rLk/edit?usp=sharing)
### Taylor Stevens and Anjali Tanna

***

## Purpose
Understanding that mitigating bias demands a multifaceted strategy anchored in data-driven methodologies, the proposed solution hinges on the creation and utilization of a meticulously curated dataset enriched with labeled information on diverse articles, revealing their inherent biases. This dataset serves as the foundation for this initiative. Central to the deployed strategy was the development and deployment of a robust multinomial regression algorithm. This advanced algorithmic framework is designed to predict and classify the bias categories of unseen articles with speed. Its predictive capabilities, powered by a wealth of annotated data, can help in taking a step in the direction of mitigating bias within the online sphere.

***

## Problem Statement
The objective of the project centers on the optimization of a multinomial regression model tailored for predicting bias in online articles. This model operates on a dataset comprising crucial columns, topic, source, bias_score, and paragraph_vectors (reflecting article content), each numerically encoded and normalized. Despite its foundation, the current model grapples with challenges pertaining to accuracy, efficiency, and interpretability. Therefore, the aim is to elevate the model's efficacy and user-friendliness, ensuring dependable predictions when confronted with unseen articles. The multinomial regression model, a statistical algorithm adept at predicting probabilities across multiple classes or categories, assumes a pivotal role in the pursuit of a working algorithm. It is finely tuned to discern and predict the bias category of online articles, drawing insights from the features inherent in the dataset columns. Through this process, the model affords nuanced probabilities to various bias categories, facilitating a comprehensive comprehension of potential biases permeating the article. By creating probability scores for each bias category, the model intricately delineates the likelihood that an article aligns with a particular bias group. These categories, left, center, or right, contingent upon dataset labels, form the foundation for the model's output, culminating in an accuracy score. This score illuminates the dominant bias ingrained within the article based on the provided features. Crucially, the model's output is engineered to be interpretable, allowing for transparency in understanding the factors of bias predictions.

***

## Dataset
The dataset `Article-Bias-Prediction.csv` utilized for training and testing the model is sourced from a publicly available repository on Google Datasets, accessible through the [Hugging Face platform.](https://huggingface.co/datasets/cjziems/Article-Bias-Prediction)

### Utilizing the `dataPreProcessing.py` File:
To create the normalized datasets, `normalizedDataNoCenter.csv`, which inlcudes only left and right biases, and `normalizedDataWithCenter.csv`, which includes left, center, and right biases, please refer to these notes.

- The code following the `FAILED DATA ATTEMPTS` section is just for reference to record our prior attempts at turning the article content data into numerical data. Please remove the two sections, `TF-IDF` and `WORD2VEC` in order to create the two datasets needed for our Multinomial Regression Model.
- Use the `Article-Bias-Prediction.csv` to:

      1. Clean the data file
      2. Perform Doc2Vec vectorization
          - This step will leave you with a file called `doc2vec_bias.csv`
      3. Normalize the `paragraph_vector` data

- Following these steps, you will have created `normalizedDataNoCenter.csv` and `normalizedDataWithCenter.csv` which can then be utilized on the Multinomial Regression Model.

## Multilogistic Regression Model

While the model outputs overall accuracies and confusion matrices, for more detailed errors the following code can be added to the `bias_classification()` function within the `logisticRegressionImplementation.py` file to output a given number of misclassification articles (in this case 10): 

```
errors = 0
for i in range(1000):
      x, y = train_data.get_sample()
      expected, actual = train_model.predict(x), y
      if (expected != actual):
          errors += 1
          print('VECTOR: ', x)
          if errors > 9:
              break
```
  
