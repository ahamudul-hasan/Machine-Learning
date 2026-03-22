# 🎯 Univariate Imputation Techniques for Handling Missing Data

## 📚 What This Folder Teaches

This folder contains **5 comprehensive tutorials** on handling missing values using **Univariate Imputation** methods. These are simple techniques that fill missing values in one feature based only on that feature's own data (not considering relationships with other features).

> **Key Concept**: Univariate = Looking at one variable at a time, independently

---

## 📖 Table of Contents

1. [Understanding Missing Data](#-understanding-missing-data)
2. [When to Use Univariate Imputation](#-when-to-use-univariate-imputation)
3. [The 5 Techniques Explained](#-the-5-techniques-explained)
4. [Files Overview](#-files-overview)
5. [Which Method to Use?](#-which-method-to-use)
6. [Real-World Scenarios](#-real-world-scenarios)

---

## 🎯 Understanding Missing Data

### What Are Missing Values?

Missing values (represented as `NaN`, `NULL`, or blank) happen when data is not recorded or available for some records. 

**Real-world causes:**
- Form submission errors
- Equipment failures
- Survey respondents skipping questions
- Data corruption during transfer
- Privacy-related omissions

### Why Should We Care?

❌ **Problems with missing data:**
- Machine learning models crash or ignore rows
- Results become biased and unreliable
- Statistical calculations fail
- Model performance drops

✅ **Solution:** Handle missing values before building models

---

## 🚀 When to Use Univariate Imputation

Univariate methods work best when:
- Missing values are **independently distributed** (not related to other variables)
- You want a **simple, fast solution**
- You have **enough non-missing data** to estimate statistics
- The feature has a **clear statistical pattern** (like age or salary)

**NOT recommended when:**
- Missingness depends on other variables (use Multivariate instead)
- Missing data is systematic/important pattern
- Data distribution is highly skewed

---

## 🔬 The 5 Techniques Explained

### 1️⃣ **Mean/Median Imputation** (`1_Mean_median.ipynb`)

**Theory:**
- Replace missing values with the **average (mean)** or **middle value (median)** of that feature
- Mean = Sum of all values ÷ Count
- Median = Middle value when sorted

**When to use:**
- ✅ Numerical data (age, salary, temperature)
- ✅ When distribution is symmetric
- ✅ Quick, interpretable results

**Pros:**
- Very simple and fast
- Preserves sample size
- Easy to understand

**Cons:**
- ❌ Reduces variance (makes data look less spread out)
- ❌ Ignores relationships between variables
- ❌ Distorts covariance/correlation
- ❌ Mean is sensitive to outliers (use median instead)

**Example:**
```
Ages: [25, 30, NULL, 35, 40]
Mean = (25+30+35+40)/4 = 32.5
After imputation: [25, 30, 32.5, 35, 40]
```

**Visual Check:** In `1_Mean_median.ipynb`, you'll see KDE plots (density curves) showing how mean/median imputation affects data distribution compared to original values.

---

### 2️⃣ **Complete Case Analysis (CCA)** (`2_CCA.ipynb`)

**Theory:**
- **Delete entire rows** that contain ANY missing value
- Keep only "complete cases" with no missing data

**When to use:**
- ✅ Small percentage of missing data (<5%)
- ✅ Missing data is MCAR (Missing Completely At Random)
- ✅ You want absolutely no imputation

**Pros:**
- ✅ No assumptions needed
- ✅ Simple to implement: `df.dropna()`
- ✅ No artificial data introduced

**Cons:**
- ❌ Loses lots of data if many rows have any missing value
- ❌ May introduce bias if missing data is related to outcomes
- ❌ Reduces statistical power
- ❌ Can lose important patterns

**Example:**
```
Original: 1000 rows
Missing values in various columns
After CCA: Maybe only 850 rows remain = 150 rows deleted!
```

**In the notebook:** You'll see how CCA significantly reduces dataset size and potentially changes data distribution (red vs green histograms).

---

### 3️⃣ **Random Sample Imputation** (`3_Random_Sample_Imputation.ipynb`)

**Theory:**
- Fill missing values by **randomly selecting from non-missing values** of that feature
- Like drawing names from a hat

**When to use:**
- ✅ Categorical data (colors, categories)
- ✅ When you want to preserve variance
- ✅ When data has no clear central tendency

**Pros:**
- ✅ Preserves variance better than mean/median
- ✅ Maintains distribution shape
- ✅ Good for categorical variables

**Cons:**
- ❌ Introduces randomness (different results each run)
- ❌ May create unrealistic value combinations
- ❌ Can introduce artificial variance

**Example:**
```
Categories: [Red, Blue, NULL, Green, Red]
Random sample picks: Green
After imputation: [Red, Blue, Green, Green, Red]
(Each run might pick differently)
```

**In the notebook:** Shows how random imputation preserves the spread of data better than mean imputation using distribution plots and variance comparisons.

---

### 4️⃣ **Missing Indicator** (`4_Missing_Indicator.ipynb`)

**Theory:**
- **Create a binary flag** (0 or 1) for each feature showing if value was originally missing
- Then handle missing values normally (impute)
- Add these flags as new features

**Why?** The fact that data is missing might be important information!

**When to use:**
- ✅ When missingness itself is predictive
- ✅ Medical data (patients skipping certain tests might indicate disease)
- ✅ When you want to capture missing pattern

**Pros:**
- ✅ Captures information about missingness
- ✅ Can improve model predictions
- ✅ Combines benefits of imputation + deletion awareness

**Cons:**
- ❌ Creates extra features (increases complexity)
- ❌ May introduce leakage in some cases

**Example:**
```
Age values:    [25, NULL, 30, NULL, 35]
Missing flag:  [0,   1,   0,   1,   0]  ← New feature!

After imputation and adding flag:
Age:           [25, 30,  30,  30, 35]
Age_missing:   [0,  1,   0,   1,  0]
```

**In the notebook:** You'll see how adding missing indicators improves classification accuracy, showing that missingness itself contains useful information.

---

### 5️⃣ **Auto-Select Best Parameters** (`5_automatically_select_imputer_parameters.ipynb`)

**Theory:**
- Use **GridSearchCV** to automatically test different imputation strategies
- Find which imputation method (mean/median/most_frequent) works best for YOUR data
- Combined with ML pipeline for end-to-end solution

**When to use:**
- ✅ When you're unsure which strategy is best
- ✅ Production ML pipelines
- ✅ When you want data-driven decisions
- ✅ Comparing multiple imputation approaches

**Pros:**
- ✅ Finds optimal strategy automatically
- ✅ Cross-validation prevents overfitting
- ✅ Integrates imputation + modeling seamlessly

**Cons:**
- ❌ Computationally expensive
- ❌ May overfit to training data patterns

**Example Workflow:**
```
1. Create pipeline: Impute → Scale → Train Model
2. Define parameter grid:
   - Imputation strategies: ['mean', 'median', 'most_frequent']
   - Model parameters: C = [0.1, 1.0, 10, 100]
3. GridSearchCV tests all combinations: 3 × 4 = 12 experiments
4. Returns best combination with highest CV score
```

**In the notebook:** Shows a complete ML pipeline with Titanic dataset, testing multiple imputation strategies on numerical (Age, Fare) and categorical (Embarked, Sex) features.

---

## 📁 Files Overview

| File | Technique | Data Type | Key Learning |
|------|-----------|-----------|---------------|
| `1_Mean_median.ipynb` | Mean/Median | Numerical | How mean/median fills gaps; impacts on variance |
| `2_CCA.ipynb` | Complete Case Analysis | Any | Data loss from deletion; distribution changes |
| `3_Random_Sample_Imputation.ipynb` | Random Sampling | Both | Preserving variance with random selection |
| `4_Missing_Indicator.ipynb` | +Indicator | Any | Adding missingness flags as features |
| `5_automatically_select_imputer_parameters.ipynb` | GridSearchCV | Both | Automated parameter tuning with pipelines |

---

## 🎯 Which Method to Use?

```
START: You have missing data
  ↓
Is missing data < 5%?
  ├─ YES → Use COMPLETE CASE ANALYSIS (CCA)
  └─ NO ↓
    ↓
Is missingness itself informative?
  ├─ YES → Use MISSING INDICATOR + imputation
  └─ NO ↓
    ↓
Numerical data?
  ├─ YES → Use MEAN/MEDIAN imputation
  └─ NO ↓
    ↓
Categorical data?
  ├─ YES → Use RANDOM SAMPLE imputation
  └─ NO ↓
    ↓
Unsure which strategy is best?
  └─ Use GRIDSEARCHCV (notebook 5)
```

---

## 📊 Real-World Scenarios

### Scenario 1: Student Grade Dataset
**Problem:** Age column is 2% missing
**Solution:** Mean/Median imputation
**Why:** Small amount of missing, numerical data, no pattern

### Scenario 2: Survey Response Dataset
**Problem:** 40% of income field is missing (missing not random)
**Solution:** Missing Indicator + imputation
**Why:** Missingness is likely important (non-respondents differ)

### Scenario 3: Medical Records
**Problem:** Test results missing for patients who didn't take test
**Solution:** Random sample or CCA depending on test importance
**Why:** Missing has meaning (patient didn't need/want test)

### Scenario 4: Production ML System
**Problem:** Need to decide imputation strategy before deployment
**Solution:** GridSearchCV to auto-tune
**Why:** Data-driven decision, reproducible, optimized

---

## ⚙️ How to Learn From This Folder

**Step 1:** Read this README to understand theory  
**Step 2:** Examine notebooks in order (1 → 5)
**Step 3:** Run notebook code cells to see outputs  
**Step 4:** Look at plots/tables to understand impact  
**Step 5:** Try modifying notebooks with your own data

---

## 💡 Key Takeaway

|  | Univariate Imputation | Complete Case Analysis |
|---|---|---|
| **Speed** | ⚡ Fast | ⚡ Very Fast |
| **Data Retained** | 🔵 All kept | 🔴 Some lost |
| **Complexity** | 🟢 Simple | 🟢 Very Simple |
| **Best For** | Small % missing | < 5% missing |
| **Bias Risk** | Medium | High |

**Remember:** Univariate methods assume missingness is independent of other variables. For complex patterns, consider Multivariate imputation (in folder `2_Multivariate`).
- **Example**: A data entry person randomly skips some entries by mistake

### 2. **Data is Sufficiently Complete**

- ✅ The proportion of missing data is small (typically < 5%)
- You have enough complete cases to maintain statistical power
- **Rule of thumb**: At least 30-50 complete observations remain

### 3. **No Important Information Loss**

- ✅ The removed rows don't contain critical information
- The subset of complete cases represents the population reasonably well

### 4. **Independence Assumption**

- ✅ Missingness is independent of the outcome variable
- The study design doesn't systematically exclude certain groups

---

## ⚖️ Advantages and Disadvantages of CCA

### ✅ **Advantages**

| Advantage                      | Explanation                                                                            |
| ------------------------------ | -------------------------------------------------------------------------------------- |
| **Simplicity**           | Easy to understand and implement - just remove incomplete rows                         |
| **No Assumptions**       | Doesn't require complex statistical models or assumptions about missing data mechanism |
| **Valid Results**        | If MCAR assumption holds, results are unbiased                                         |
| **Preserves Data**       | No data imputation means no artificial values introduced                               |
| **Complete Information** | Working with genuine, observed data only                                               |

### ❌ **Disadvantages**

| Disadvantage                        | Explanation                                                           |
| ----------------------------------- | --------------------------------------------------------------------- |
| **Data Loss**                 | Can significantly reduce sample size, losing statistical power        |
| **Reduced Precision**         | Smaller sample = larger confidence intervals = less precise estimates |
| **Bias Risk**                 | If missing data is not MCAR, results become biased                    |
| **Inefficient**               | Wastes information that could be recovered through imputation         |
| **Not Suitable for MAR/MNAR** | Fails when missingness depends on observed or unobserved variables    |

### Visual Comparison

```
Sample Size Impact:
Original Dataset: 1000 observations
Missing Data: 20% (200 rows have at least one missing value)

After CCA: 800 observations
Data Reduction: 20% loss

Statistical Impact:
- Standard Error: Increases
- Confidence Intervals: Become wider
- Test Power: Decreases
```

---

## 🤔 When to Use CCA?

### ✅ **USE CCA When:**

1. **Small Amount of Missing Data**

   - Less than 5% of total data is missing
   - Loss of a few rows won't impact analysis
2. **Data is MCAR**

   - Missing values are truly random
   - No systematic pattern to missingness
3. **Sufficient Sample Size**

   - Even after removing incomplete cases, enough data remains
   - Statistical power is maintained
4. **Missing Data is Scattered**

   - Missing values are spread across different variables
   - Not concentrated in a few columns
5. **Data Quality is Priority**

   - You only want to work with verified, complete observations
   - Better to have less data than inaccurate conclusions

### ❌ **DON'T USE CCA When:**

- ❌ Large amount of missing data (> 10%)
- ❌ Missing data is MAR or MNAR (not random)
- ❌ Only one column has many missing values (delete column instead)
- ❌ You can't afford to lose many observations
- ❌ Missing data is correlated with the outcome variable

---

## 💻 Code Explanation

### Overview

The `1_CCA.ipynb` notebook demonstrates Complete Case Analysis using the **Data Science Job Placement** dataset. It shows how to identify missing data and apply CCA.

### Step-by-Step Code Walkthrough

#### **Step 1: Import Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

- `numpy`: Numerical operations
- `pandas`: Data manipulation and analysis
- `matplotlib`: Data visualization

---

#### **Step 2: Load Data**

```python
df = pd.read_csv('../../Data/data_science_job.csv')
df.head()
```

**What it does:**

- Loads the CSV file into a pandas DataFrame
- `head()` shows first 5 rows to inspect the data structure
- Path uses relative reference (`../../Data/`) to locate the data folder

**Output:** First few rows of the dataset

---

#### **Step 3: Assess Missing Data Percentage**

```python
(df.isnull().mean() * 100).astype(str) + ' %'
```

**What it does:**

1. `df.isnull()` - Creates a boolean DataFrame (True where values are missing)
2. `.mean()` - Calculates proportion of missing values per column
3. `* 100` - Converts to percentage
4. `.astype(str) + ' %'` - Formats as percentage string

**Example Output:**

```
age                    0.0 %
salary                 2.5 %
experience             1.8 %
education_level        3.2 %
...
```

---

#### **Step 4: Check Dataset Dimensions**

```python
df.shape
```

**What it does:**

- Returns tuple (number of rows, number of columns)
- Helps understand dataset size before and after CCA

**Example Output:** `(19158, 14)` - 19,158 rows and 14 columns

---

#### **Step 5: Identify Columns with Missing Data**

```python
cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 
        and df[var].isnull().mean() > 0]
cols
```

**What it does:**

1. Filters columns that have:
   - Missing data > 0% (at least some missing values)
   - Missing data < 5% (not too much missing)
2. Creates a list of these column names

**Why:** Focuses on columns where CCA is appropriate

**Example Output:** `['salary', 'experience', 'education_level']`

---

#### **Step 6: Sample Missing Data**

```python
df[cols].sample(5)
```

**What it does:**

- Shows 5 random rows from columns with missing data
- Helps visualize the missing values

**Example Output:**

```
      salary    experience    education_level
12845   NULL      3           Bachelors
23456   50000     NULL        Masters
...
```

---

#### **Step 7: Check Category Distribution**

```python
df['education_level'].value_counts()
```

**What it does:**

- Shows frequency of each category in the column
- Helps understand data characteristics

**Example Output:**

```
Bachelors      8500
Masters        5200
High School    2100
PhD            1600
```

---

#### **Step 8: Calculate Data Retention Rate**

```python
len(df[cols].dropna()) / len(df)
```

**What it does:**

1. `df[cols].dropna()` - Removes rows with any missing in selected columns
2. `len()` - Counts remaining rows
3. Divides by total to get retention rate

**Example Output:** `0.92` (92% of data retained)

---

#### **Step 9: Apply Complete Case Analysis**

```python
new_df = df[cols].dropna()
df.shape, new_df.shape
```

**What it does:**

1. `.dropna()` - **Removes all rows with any missing values**
2. Creates new dataset with only complete cases
3. Compares shapes before and after

**Example Output:**

```
Original: (19158, 3)
After CCA: (17585, 3)
Rows Removed: 1573
```

---

#### **Step 10: Visualize Data Distributions**

```python
new_df.hist(bins=50, density=True, figsize=(12, 12))
plt.show()
```

**What it does:**

- Creates histograms for all numerical columns
- `bins=50` - Divides data into 50 bars
- `density=True` - Normalizes to probability density
- `figsize=(12,12)` - Sets large display size

**Purpose:** Shows distribution patterns in the clean data

---

#### **Step 11: Compare Before/After Distributions**

```python
fig = plt.figure()
ax = fig.add_subplot(111)

# Original data
df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')

# Data after CCA
new_df['training_hours'].hist(bins=50, ax=ax, color='green', 
                              density=True, alpha=0.8)
```

**What it does:**

1. Creates overlaid histograms
2. **Red** = Original dataset (with missing values)
3. **Green** = Dataset after CCA (complete cases only)
4. `alpha=0.8` - Makes green slightly transparent to see overlap

**Why:** Shows if CCA changes the data distribution (potential bias indicator)

---

#### **Step 12: Density Plots**

```python
fig = plt.figure()
ax = fig.add_subplot(111)

# Original data
df['training_hours'].plot.density(color='red')

# Data after CCA
new_df['training_hours'].plot.density(color='green')
```

**What it does:**

- Creates smooth distribution curves instead of histograms
- Easier to compare general trend shapes

**Interpretation:**

- If red and green curves are similar → CCA didn't introduce bias ✓
- If they differ significantly → Missing data was not MCAR ⚠️

---

#### **Step 13: Compare Categorical Variables**

```python
temp = pd.concat([
    df['education_level'].value_counts() / len(df),
    new_df['education_level'].value_counts() / len(new_df)
], axis=1)

temp.columns = ['original', 'cca']
temp
```

**What it does:**

1. Creates proportion distributions for before/after CCA
2. `value_counts()` - Counts each category
3. Divides by total length for proportions
4. Side-by-side comparison in table format

**Example Output:**

```
                original    cca
Bachelors         0.445    0.444
Masters           0.272    0.273
High School       0.110    0.109
PhD               0.084    0.084
```

**Interpretation:**

- If proportions are nearly identical → CCA maintained data structure ✓
- If they differ → Different groups have different missing rates ⚠️

---

## 🎓 Key Takeaways

| Concept         | Summary                                        |
| --------------- | ---------------------------------------------- |
| **CCA**   | Remove rows with any missing values            |
| **When**  | Use when <5% missing and MCAR assumption holds |
| **Pros**  | Simple, no bias if MCAR                        |
| **Cons**  | Loss of data, reduced statistical power        |
| **Check** | Verify distributions before vs. after CCA      |

---

## 📚 Additional Resources

- **Missing Data Types**: MCAR, MAR, MNAR
- **Alternative Methods**: Imputation (Mean, KNN, MICE)
- **Best Practice**: Always explore why data is missing before deciding on CCA

---

**Created for Machine Learning Data Engineering Module** ✨
