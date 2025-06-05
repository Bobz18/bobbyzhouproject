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
    src = "rating-distribution.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

This bias in ratings could affect model predictions, as the model may overfit to highly rated recipes.

---

### Bivariate Analysis

To assess relationships between variables, I explored whether recipes tagged with “meat” received different ratings compared to those without. Using a kernel density estimate (KDE) plot, I found that the inclusion of meat tags does not appear to significantly impact average recipe rating.

<iframe
    src = "assets/bivariate_1.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

This analysis helped inform my choice of features for model development and fairness analysis later on.

---

### Interesting Aggregates

I also computed summary statistics for several numerical features by rating category. The table below shows the average `minutes`, `n_steps`, and `n_ingredients` for each rating level from 1 to 5:

| rating | minutes | n_steps | n_ingredients |
|--------|---------|---------|----------------|
| 1      | 38.72   | 10.33   | 8.80           |
| 2      | 39.63   | 10.40   | 9.13           |
| 3      | 38.09   | 9.74    | 9.10           |
| 4      | 36.47   | 9.43    | 9.00           |
| 5      | 36.36   | 9.71    | 8.95           |

This pivot table suggests a possible trend that recipes with higher ratings may tend to be simpler (require less time or fewer steps/ingredients), though the differences are subtle. These insights helped inform which variables to prioritize in modeling stages.


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
    src = "graphs/MAR_n_steps.html"
    width = "800"
    height = "600"
    frameborder = "0"
    style="margin: 0; padding: 0; display: block;"
></iframe>

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
