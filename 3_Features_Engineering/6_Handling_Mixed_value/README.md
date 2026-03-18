# 📊 Handling Missing Data: Complete Case Analysis (CCA)

---

## 📑 Table of Contents
1. [Handling Missing Data](#-handling-missing-data)
2. [Complete Case Analysis (CCA)](#-complete-case-analysis-cca)
3. [Assumptions for CCA](#-assumptions-for-cca)
4. [Advantages and Disadvantages](#-advantages-and-disadvantages-of-cca)
5. [When to Use CCA?](#-when-to-use-cca)
6. [Code Explanation](#-code-explanation)

---

## 🎯 Handling Missing Data

### What is Missing Data?

Missing data refers to the absence of values for certain observations in a dataset. In real-world datasets, it's common to encounter incomplete information due to various reasons:

- **Data collection errors** - Information not recorded properly
- **Equipment malfunction** - Data not captured due to technical issues
- **Privacy concerns** - Sensitive information intentionally not recorded
- **User non-response** - Survey respondents skipping questions
- **Data loss** - Information lost during storage or transfer

### Why is it Important?

Missing data can:
- ❌ Reduce the statistical power of analyses
- ❌ Introduce bias in results
- ❌ Lead to misleading conclusions
- ❌ Cause machine learning models to perform poorly

### Common Approaches to Handle Missing Data

| Approach | Description | Best For |
|----------|-------------|----------|
| **Deletion** | Remove rows/columns with missing values | Small amount of missing data |
| **Imputation** | Fill missing values with estimates | When data is MCAR or MAR |
| **Prediction** | Use models to predict missing values | Complex relationships in data |
| **Domain Knowledge** | Fill based on expert judgment | Critical values with context |

---

## 🔍 Complete Case Analysis (CCA)

### What is Complete Case Analysis?

**Complete Case Analysis (CCA)**, also known as **Listwise Deletion**, is a method where:

> 📌 **We remove any row (observation) that contains at least one missing value**

After CCA, we're left with a dataset containing only complete cases - rows with no missing values.

### How Does CCA Work?

```
Original Dataset:
┌─────────────────────────────────────┐
│ ID │ Age │ Salary │ Experience │    │
├────┼─────┼────────┼────────────┤    │
│ 1  │ 25  │ 50000  │ 3          │ ✓  │ Complete
│ 2  │ 28  │ NULL   │ 5          │ ✗  │ Has missing
│ 3  │ 32  │ 75000  │ 8          │ ✓  │ Complete
│ 4  │ NULL│ 60000  │ 4          │ ✗  │ Has missing
│ 5  │ 30  │ 65000  │ 6          │ ✓  │ Complete
└─────────────────────────────────────┘

After CCA:
┌─────────────────────────────────────┐
│ ID │ Age │ Salary │ Experience │    │
├────┼─────┼────────┼────────────┤    │
│ 1  │ 25  │ 50000  │ 3          │ ✓  │
│ 3  │ 32  │ 75000  │ 8          │ ✓  │
│ 5  │ 30  │ 65000  │ 6          │ ✓  │
└─────────────────────────────────────┘
```

### Implementation in Python

```python
# Simple CCA: Remove all rows with any missing values
clean_data = data.dropna()

# Selective CCA: Remove rows with missing in specific columns
clean_data = data.dropna(subset=['column1', 'column2'])
```

---

## 📋 Assumptions for CCA

### 1. **MCAR: Missing Completely at Random**
- ✅ Missing values are completely random
- The probability of being missing is independent of any variable
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

| Advantage | Explanation |
|-----------|-------------|
| **Simplicity** | Easy to understand and implement - just remove incomplete rows |
| **No Assumptions** | Doesn't require complex statistical models or assumptions about missing data mechanism |
| **Valid Results** | If MCAR assumption holds, results are unbiased |
| **Preserves Data** | No data imputation means no artificial values introduced |
| **Complete Information** | Working with genuine, observed data only |

### ❌ **Disadvantages**

| Disadvantage | Explanation |
|--------------|-------------|
| **Data Loss** | Can significantly reduce sample size, losing statistical power |
| **Reduced Precision** | Smaller sample = larger confidence intervals = less precise estimates |
| **Bias Risk** | If missing data is not MCAR, results become biased |
| **Inefficient** | Wastes information that could be recovered through imputation |
| **Not Suitable for MAR/MNAR** | Fails when missingness depends on observed or unobserved variables |

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
df = pd.read_csv('../../CSV/data_science_job.csv')
df.head()
```

**What it does:**
- Loads the CSV file into a pandas DataFrame
- `head()` shows first 5 rows to inspect the data structure
- Path uses relative reference (`../../CSV/`) to locate the data folder

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

| Concept | Summary |
|---------|---------|
| **CCA** | Remove rows with any missing values |
| **When** | Use when <5% missing and MCAR assumption holds |
| **Pros** | Simple, no bias if MCAR |
| **Cons** | Loss of data, reduced statistical power |
| **Check** | Verify distributions before vs. after CCA |

---

## 📚 Additional Resources

- **Missing Data Types**: MCAR, MAR, MNAR
- **Alternative Methods**: Imputation (Mean, KNN, MICE)
- **Best Practice**: Always explore why data is missing before deciding on CCA

---

**Created for Machine Learning Data Engineering Module** ✨
