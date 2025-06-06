# Recipe Rating Predictions Based on Meat Inclusion
Author: Bobby Zhou

## Overview
This data science project is the final projects for DSC80-2025-sp at UCSD. It is focused on predicting the recipe's rating based on relevant info.

## Introduction
In recent years, dietary preferences have shifted significantly, with growing attention to how recipe composition affects consumer perceptions and ratings. In particular, the presence of meat in recipes has become a key focus, as the rise of plant-based eating reflects increasing awareness of health and environmental concerns.

In this project, I examined two tables:

1. `RAW_recipes.csv` contains all recipes
2. `RAW_interactions.csv` contains all reviews and ratings submitted for recipes in `RAW_interactions.csv`

This dataset offers a comprehensive variety of recipes, spanning different cuisines, levels of difficulty, and ingredient types, making it well-suited for analyzing recipe quality across diverse cooking contexts. My goal is to investigate what factors most strongly influence a recipe’s rating, and in doing so, gain insight into broader consumer preferences.

I will begin by cleaning the data and conducting an exploratory analysis to identify key patterns and potential issues.
Next, I will examine the nature of missing data in the review column to understand whether the absence of values follows a particular pattern.

**In the final stage, I will develop a predictive model that estimates recipe ratings using selected features from the dataset.**

The first dataset, `recipe`, that is coming from `RAW_recipe.csv` contains 83782 rows and 12 columns, each row is a unique recipe. The columns are:

| **Column**       | **Description**                                                                                                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`           | Recipe name                                                                                                                                                                                         |
| `id`             | Recipe ID                                                                                                                                                                                           |
| `minutes`        | Minutes to prepare recipe                                                                                                                                                                           |
| `contributor_id` | User ID who submitted this recipe                                                                                                                                                                   |
| `submitted`      | Date recipe was submitted                                                                                                                                                                           |
| `tags`           | Food.com tags for recipe                                                                                                                                                                            |
| `nutrition`      | Nutrition information in the form `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`; PDV stands for “percentage of daily value” |
| `n_steps`        | Number of steps in recipe                                                                                                                                                                           |
| `steps`          | Text for recipe steps, in order                                                                                                                                                                     |
| `description`    | User-provided description                                                                                                                                                                           |
| `ingredients`    | Ingredients of the recipe                                                                                                                                                                           |
| `n_ingredients`  | Number of ingredients for the recipe                                                                                                                                                                |

The second dataset, `interactions`, contains 731927 rows and 5 columns, each row correspond to a review to a recipe in `recipe`. The columns are:

| **Column**  | **Description**     |
| ----------- | ------------------- |
| `user_id`   | User ID             |
| `recipe_id` | Recipe ID           |
| `date`      | Date of interaction |
| `rating`    | Rating given        |
| `review`    | Review text         |

## Data Cleaning and Exploratory Analysis

### Data Cleaning

To prepare the data for analysis, I performed several cleaning operations on the merged dataset:

1. **Merged Datasets**: Performed a left join on `RAW_recipes.csv` and `RAW_interactions.csv` using the recipe ID to combine recipe information with its corresponding reviews.
2. **Dropped Invalid Rows**: Removed one row where the recipe name was missing (`NaN`) and all rows with missing `user_id` values, as these rows did not have associated reviews.
3. **Type Conversions**:
   - Converted `submitted` and `date` columns to `datetime` format to allow time-based feature extraction later.
   - Converted columns like `tags`, `nutrition`, and `ingredients` from string representations to Python lists.
4. **Imputed Ratings**: Replaced `rating == 0` with `NaN`, as a rating of 0 is considered missing (valid ratings range from 1 to 5).
5. **Added Feature**: 
   - Created a new column `average_rating` that stores the average rating for each recipe across all reviews.
   - Created a new column `contains_meat` that stores the boolean value for each recipe that indicates a recipe's meat inclusion status.

These steps were necessary to ensure that the analysis would not be biased by incorrect or incomplete values, and that each variable was in a usable format for subsequent processing.

### Results

The final DataFrame contains the following columns:

| **Column**       | **Dtype**      |
| ---------------- | -------------- |
| `name`           | object         |
| `id`             | int64          |
| `minutes`        | int64          |
| `contributor_id` | int64          |
| `submitted`      | datetime64[ns] |
| `tags`           | object         |
| `nutrition`      | object         |
| `n_steps`        | int64          |
| `steps`          | object         |
| `description`    | object         |
| `ingredients`    | object         |
| `n_ingredients`  | int64          |
| `user_id`        | float64        |
| `recipe_id`      | float64        |
| `date`           | datetime64[ns] |
| `rating`         | float64        |
| `review`         | object         |
| `average_rating` | float64        |
| `contains_meat`  | bool           |


I display the head of the cleaned DataFrame below:

| name                               | id      | minutes | submitted           | rating | avg_rating | ... |
|------------------------------------|---------|---------|---------------------|--------|------------|-----|
| 1 brownies in the world best ever  | 333281  | 40      | 2008-10-27          | 4      | 4.0        | ... |
| 1 in canada chocolate chip cookies | 453467  | 45      | 2011-04-11          | 5      | 5.0        | ... |

---

### Univariate Analysis

I examined the distribution of `rating` values across the dataset. The histogram reveals that the ratings are heavily skewed to the right, with most reviews being 4 or 5 stars. This suggests that users tend to leave reviews only when they have strong (mostly positive) experiences with recipes.

<iframe
    src = "graphs/rating-distribution.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

This bias in ratings could affect model predictions, as the model may overfit to highly rated recipes.

---

### Recipe Preparation Time Distribution

To understand how long recipes typically take to prepare, I plotted the distribution of preparation time (in minutes), removing extreme outliers using IQR filtering. The histogram below shows that most recipes fall between 0 and 60 minutes, with a sharp drop in frequency after that point. A large concentration of recipes take between 10 to 40 minutes to make, and the distribution is clearly right-skewed.

<iframe
    src="graphs/minutes_distribution.html"
    width="800"
    height="600"
    frameborder="0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

This pattern suggests that most users prefer quicker recipes that can be completed within an hour — likely due to time constraints or everyday convenience. Recipes requiring longer preparation times are far less common, possibly because they cater to niche or special-occasion cooking. The skewness in time may also affect model behavior, favoring faster recipes as they dominate the dataset.

---

### Bivariate Analysis

To assess relationships between variables, I explored whether recipes tagged with “meat” received different ratings compared to those without. Using a kernel density estimate (KDE) plot, I found that the inclusion of meat tags does not appear to significantly impact average recipe rating.

<iframe
    src = "graphs/.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

This analysis helped inform my choice of features for model development and fairness analysis later on.

---

### Complexity and Time vs. Recipe Rating (3D Plot)

To explore how recipe complexity and preparation time relate to user ratings, I created a 3D scatter plot using `minutes`, `n_ingredients`, and `n_steps`, with color representing the `rating`. Each point represents a single recipe, and outliers in preparation time were removed using IQR filtering for clarity.

<iframe
    src="graphs/3D-rating-distribution.html"
    width="900"
    height="700"
    frameborder="0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

The plot reveals that high-rated recipes (shown in yellow) span across a wide range of time and complexity values, but tend to cluster more densely around moderate preparation times and lower ingredient/step counts. This suggests that while users may reward recipes that are relatively simple and quick, highly rated dishes are not restricted to minimal effort — they may also include thoughtfully crafted, time-intensive options that deliver high satisfaction. There is no sharp linear relationship, implying that other factors beyond time and complexity (such as taste or clarity of instructions) likely influence ratings.


### Interesting Aggregates
#### Preparation Time and Complexity by Rating

To examine how recipe complexity and preparation time relate to ratings, I computed average values for `minutes`, `n_steps`, and `n_ingredients` grouped by rating level. The results are shown below:

| Rating | Minutes | Steps | Ingredients |
|--------|---------|--------|-------------|
| 1      | 99.67   | 10.63  | 8.91        |
| 2      | 98.02   | 10.70  | 9.23        |
| 3      | 87.50   | 9.99   | 9.20        |
| 4      | 91.59   | 9.58   | 9.10        |
| 5      | 106.92  | 9.98   | 9.05        |


The complexity-related metrics (`minutes`, `n_steps`, and `n_ingredients`) reveal interesting trends across rating levels. From ratings 1 to 3, there is a general decrease in both preparation time and number of steps, which may suggest that simpler or quicker recipes are slightly more favored. Rating 4 follows this pattern with the lowest average step count, but at rating 5, the average preparation time spikes noticeably, while step and ingredient counts remain moderate.

This pattern may indicate that while users appreciate simplicity, the highest-rated recipes are not necessarily the fastest or easiest to make. Instead, they might strike a desirable balance between effort and reward—requiring more time but not necessarily more steps or complexity. The uptick in time at rating 5 suggests that top-rated recipes may emphasize thoughtful preparation or longer cook times that enhance flavor or texture, even if the ingredient count remains stable.

#### Recipe Characteristics by Rating

To better understand how recipe characteristics vary by rating, I computed average values for several nutritional and time-based features grouped by rating. The table below summarizes the mean `minutes`, `calories`, `total_fat`, `sugar`, `protein`, `saturated_fat`, and `carbohydrates` for recipes rated from 1 to 5:

| Rating | Calories | Carbohydrates | Minutes | Protein | Saturated Fat | Sugar  | Total Fat |
|--------|----------|----------------|---------|---------|----------------|--------|------------|
| 1      | 486.60   | 16.40          | 99.67   | 34.06   | 46.68          | 88.01  | 37.06      |
| 2      | 446.60   | 15.04          | 98.02   | 34.31   | 42.88          | 75.17  | 32.77      |
| 3      | 425.79   | 13.68          | 87.45   | 34.86   | 40.09          | 65.55  | 31.64      |
| 4      | 405.04   | 12.83          | 91.59   | 34.05   | 36.43          | 56.79  | 29.94      |
| 5      | 415.21   | 13.04          | 106.92  | 32.64   | 39.23          | 63.08  | 31.79      |

Overall, the nutritional metrics (`calories`, `carbohydrates`, `protein`, `saturated_fat`, `sugar`, and `total_fat`) show a fairly clear downward trend as the rating increases from 1 to 4. Recipes with lower ratings tend to have higher energy content and richer nutritional profiles — for example, average calories drop from about 486 at rating 1 to around 405 at rating 4, along with noticeable declines in sugar, fat, and carbohydrates.

This suggests that users may associate lower-calorie, lower-sugar recipes with higher quality or satisfaction, potentially reflecting health-conscious preferences. However, at rating 5, some metrics such as `minutes`, `saturated_fat`, and `carbohydrates` tick upward slightly. This reversal could imply that the highest-rated recipes balance nutrition with enhanced flavor or complexity — they may be more indulgent, take longer to prepare, or use richer ingredients to achieve superior taste.

In short, while there’s a general trend toward healthier profiles as rating increases, rating 5 recipes may reflect a nuanced preference: dishes that are worth the extra time or richness for a top-tier experience.


## Missingness Mechanism

The three columns in the cleaned DataFrame with a non-trivial amount of missing values are `rating` (**15,036** missing entries), `review` (**2,777**), and `description` (**136**).

### NMAR Analysis

I do not believe that any of these columns are **Not Missing at Random (NMAR)**. In fact, all three are likely **Missing at Random (MAR)**, as their missingness appears to be associated with other observable features in the dataset.

For example, users who enjoyed a recipe are more likely to provide a positive `rating`, leave a detailed `review`, or even write a `description`. The presence or absence of these entries could also depend on attributes such as the `contributor_id` (reflecting how active a user is) or the `name` of the recipe (since certain recipes may be seasonal, holiday-specific, or uniquely memorable, attracting more engagement).

In particular, I believe that the missingness of `rating` is most likely influenced by other columns and is not random. This justifies further investigation into the dependency of `rating` on other features in the dataset.


### Missingness Dependency

Since my goal is to predict `rating`, I began by examining whether its missingness depends on other features in the dataset. To explore this, I referred back to the 3D scatter plot I created in Step 2, which visualizes `minutes`, `n_steps`, and `n_ingredients`, with points colored by `rating`. This visualization helped me inspect whether recipes that are more time-consuming or complex tend to have missing or extreme `rating` values. Thus, I wanted to know if the absence of `rating` is conditionally related to `n_steps` (the number of steps in a recipe), `minutes` (the preparation time), or `n_ingredients` (a proxy for complexity).

> `n_steps` and `rating`

I suspect that users may be less likely to leave a rating for recipes that involve too many steps, perhaps due to fatigue or a lower likelihood of completing the recipe.

**Null Hypothesis (H₀)**: The missingness of `rating` is independent of `n_steps`.

**Alternative Hypothesis (H₁)**: The missingness of `rating` depends on `n_steps`.

**Test Statistic**: The difference in the average number of steps (`n_steps`) between the group where `rating` is missing and the group where `rating` is not missing.

**Significance Level**: 0.01

I conducted a permutation test by randomly shuffling the missingness indicator for `rating` 1,000 times. The resulting distribution of the test statistic, along with the observed value, is shown below:

<iframe 
    src="graphs/MAR_n_steps.html" 
    width="800" 
    height="600" 
    frameborder="0" 
    style="margin: 0; padding: 0; display: block;">
</iframe>

The p-value was approximately 0.0, which is below the significance level of 0.01. Therefore, I reject the null hypothesis and conclude that the missingness of `rating` is **MAR**, conditional on `n_steps`.

> `n_ingredients` and `rating`

I also considered whether the number of ingredients in a recipe affects the likelihood of a user submitting a rating. It’s possible that more complex recipes (those requiring more ingredients) may discourage users from leaving feedback.

**Null Hypothesis (H₀)**: The missingness of `rating` is independent of `n_ingredients`.

**Alternative Hypothesis (H₁)**: The missingness of `rating` depends on `n_ingredients`.

**Test Statistic**: The difference in the average number of ingredients (`n_ingredients`) between the group where `rating` is missing and the group where `rating` is not missing.

**Significance Level**: 0.01

I conducted a permutation test by randomly shuffling the missingness indicator for `rating` 1,000 times. The resulting distribution of the test statistic, along with the observed value, is shown below:

<iframe
    src = "graphs/MAR_n_ingredients.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

The p-value was approximately 0.0, which is below the significance level of 0.01. Therefore, I reject the null hypothesis and conclude that the missingness of `rating` is **MAR**, conditional on `n_ingredients`.

> `minutes` and `rating`

Finally, I investigated whether the time it takes to prepare a recipe influences the likelihood of a user providing a rating. It seems plausible that users may be less motivated to leave feedback after spending a long time cooking.

**Null Hypothesis (H₀)**: The missingness of `rating` is independent of `minutes`.

**Alternative Hypothesis (H₁)**: The missingness of `rating` depends on `minutes`.

**Test Statistic**: The difference in the average preparation time (`minutes`) between the group where `rating` is missing and the group where `rating` is not missing.

**Significance Level**: 0.01

I conducted a permutation test by randomly shuffling the missingness indicator for `rating` 1,000 times. The resulting distribution of the test statistic, along with the observed value, is shown below:

<iframe
    src = "graphs/MAR_minutes.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

The p-value was approximately 0.139, which is above the significance level of 0.01. Therefore, I fail to reject the null hypothesis and conclude that the missingness of `rating` is **not** dependent on `minutes`.

## Step 4: Hypothesis Testing

As mentioned earlier in my exploratory analysis, I observed that recipes containing meat might be rated differently from those that do not. To formally test whether this difference is statistically significant, I performed a permutation test comparing the average ratings between meat and non-meat recipes.

### Hypothesis

I defined a binary column called `contains_meat`, which indicates whether the word `"meat"` appears in the `tags` for a recipe.

I then conducted a hypothesis test to examine whether the average rating differs between recipes that contain meat and those that do not.

**Null Hypothesis (H₀)**: The average rating is the same for recipes with and without meat tags. Any observed difference is due to random chance.

**Alternative Hypothesis (H₁)**: The average rating for recipes without meat tags is higher than that for recipes with meat tags.

**Test Statistic**: The difference in mean rating (`mean_rating_no_meat - mean_rating_meat`).

**Significance Level**: 0.05

### Procedure

I randomly shuffled the `contains_meat` column 10,000 times to simulate the null distribution of the test statistic. For each permutation, I computed the difference in average ratings between the two groups. I then compared the observed statistic to this distribution to compute a p-value.

### Result

The observed difference in means was positive, indicating that meatless recipes tended to have higher ratings. The p-value from the permutation test was approximately **0.0**, which is below the 0.05 significance threshold.

### Conclusion

Since I get p-value = 0.0 < 0.05, I reject the null hypothesis and conclude that recipes without meat tags tend to receive higher ratings than those with meat tags. This suggests that meat inclusion may negatively influence how recipes are rated by users.

## Framing a Prediction

Based on my earlier analysis, I aim to **predict the rating of a recipe** as a multi-class classification problem. Since `rating` takes on integer values from 1 to 5, I treat it as a categorical, ordinal variable.

I chose `rating` as the response variable because it is the most visible and influential metric that users consider when browsing recipes online. Additionally, most of my prior exploratory analysis centered around understanding what affects `rating`, giving me a strong foundation for identifying relevant features.

In particular, I found that both `minutes` (preparation time) and `contains_meat` (meat inclusion) were significantly associated with recipe ratings. Therefore, I will use these two variables to construct features for my baseline classification model.

To assess the performance of the model, I will evaluate two metrics:

1. **Accuracy**  
   Accuracy measures the overall proportion of correct predictions. It provides a high-level indication of how well the model predicts ratings using `minutes` and `contains_meat` as inputs.

2. **F1 Score**  
   The F1 score balances precision and recall, making it useful for understanding how well the model performs across all rating categories—especially in the presence of class imbalance. It also helps assess whether the model is generating excessive false positives or false negatives, which is important for evaluating fairness and reliability.


## Baseline Model

For my baseline model, I used a **Random Forest Classifier** to predict recipe ratings based on two key features: `minutes` and `contains_meat`. These features were chosen based on earlier exploratory analysis that indicated their relevance to rating outcomes.

To prepare the data:

1. **`minutes`**  
   This is a continuous numerical feature representing the preparation time of a recipe. I standardized this column using `StandardScaler()` so that the model could interpret whether a recipe takes more or less time than average, regardless of scale.

2. **`contains_meat`**  
   This is a boolean feature indicating whether a recipe includes meat. Since prior analysis showed meaningful rating differences between meat and non-meat recipes, I retained this variable in data cleaning steps and converted it into an integer (1 for `True`, 0 for `False`) using a `FunctionTransformer`.

I did not tune any hyperparameters — the Random Forest was run with default settings. The dataset was split using `train_test_split` with 80% of data used for training and 20% held out for testing.

---

### Evaluation Results

Here are the performance metrics from the baseline model:

- **Accuracy:** `0.776`
- **Macro F1-Score:** `0.175`
- **Per-Class F1-Scores:** `[0.0, 0.0, 0.0, 0.0, 0.87]`

Despite the relatively high accuracy (about 77.6%), the **F1-score reveals a critical issue**: the model performs well only for recipes rated 5, while it completely fails to predict ratings 1 through 4. The high accuracy is misleading, as it is driven by the class imbalance in the dataset — around 95% of recipes have ratings of 4 or 5.

Upon inspecting the predictions, I found that the model mostly outputs only 4s and 5s. This reflects the skewed distribution of ratings and explains the poor macro F1-score. The model, while technically "accurate", is not fair or informative across rating categories.

---

### Cross-Validation Results

To better evaluate the reliability of my baseline model, I performed **5-fold cross-validation** using both accuracy and F1 score as evaluation metrics. This technique helps assess the model’s performance across different subsets of the data and provides a more robust estimate of generalization.

Here are the averaged results across the five folds:

- **Mean Accuracy:** `0.7753`
- **Mean F1 Score (weighted):** `0.6773`

These results are consistent with the earlier train-test split, reaffirming that the model maintains high accuracy. However, the **weighted F1 score** is significantly higher than the earlier macro F1 score (which was only 0.175). This is because the **weighted F1 score gives more importance to dominant classes** (like ratings 4 and 5), whereas the macro score weighs each class equally.

In short, while the model appears effective based on overall accuracy and weighted F1, it still fails to meaningfully capture the minority classes (ratings 1–3), which is masked by the imbalanced label distribution. This reinforces the need to address class imbalance and refine evaluation strategies in future modeling stages.

---

### Next Steps

As it stands, this baseline model is not practically useful. It performs similarly to a naive classifier that always predicts 5. To improve the model, I will need to address the **class imbalance** and enhance the model’s ability to distinguish lower-rated recipes. This will be a priority in the final modeling stage.


