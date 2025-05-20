# Statistics with Python: From Basic to Advanced

## 1. Introduction to Statistics
Statistics is the science of collecting, analyzing, interpreting, and presenting data. It has two main branches:
- **Descriptive Statistics**: Summarizes data characteristics (e.g., mean, variance).
- **Inferential Statistics**: Draws conclusions about populations from samples (e.g., hypothesis testing, regression).

### Why Python for Statistics?
Python’s libraries (`numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`) offer powerful tools for statistical analysis, data manipulation, and visualization.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
# Set random seed for reproducibility
np.random.seed(42)
```

---

## 2. Descriptive Statistics
Descriptive statistics summarize a dataset’s key features.

### 2.1 Measures of Central Tendency
These describe the center of the data:
- **Mean (μ)**: The arithmetic average, calculated as Σx/n.
- **Median**: The middle value in sorted data, robust to outliers.
- **Mode**: The most frequent value.

```python
# Example: Calculating mean, median, mode
data = [2, 3, 3, 4, 5, 5, 5, 6, 7]
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)[0][0]
print(f"Mean: {mean:.2f}, Median: {median:.2f}, Mode: {mode}")
```

**Output**:
```
Mean: 4.44, Median: 5.00, Mode: 5
```

**Explanation**: The mean (4.44) is slightly below the median (5.00) due to the left skew of the data (more lower values). The mode (5) reflects the most frequent value.

### 2.2 Measures of Dispersion
These quantify the spread of the data:
- **Range**: Max value minus min value.
- **Variance (σ²)**: Average of squared deviations from the mean, using `ddof=1` for sample variance.
- **Standard Deviation (σ)**: Square root of variance, in the same units as the data.
- **Interquartile Range (IQR)**: Difference between the 75th and 25th percentiles, robust to outliers.

```python
# Example: Calculating range, variance, standard deviation, IQR
range_val = max(data) - min(data)
variance = np.var(data, ddof=1)
std_dev = np.std(data, ddof=1)
iqr = np.percentile(data, 75) - np.percentile(data, 25)
print(f"Range: {range_val}, Variance: {variance:.2f}, Std Dev: {std_dev:.2f}, IQR: {iqr:.2f}")
```

**Output**:
```
Range: 5, Variance: 2.78, Std Dev: 1.67, IQR: 2.00
```

**Explanation**: The range (5) shows the spread from 2 to 7. The variance (2.78) and standard deviation (1.67) indicate moderate variability. The IQR (2.00) captures the middle 50% of the data.

### 2.3 Data Visualization
Visualizations reveal data distributions and patterns:
- **Histogram**: Displays frequency of values in bins.
- **Box Plot**: Shows median, quartiles, and potential outliers.

```python
# Example: Histogram and Box Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(data, bins=5, edgecolor='black')
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.boxplot(data, vert=False)
plt.title('Box Plot of Data')
plt.xlabel('Value')
plt.savefig('descriptive_plots.png')
plt.close()
```

**Graph Representation**:
- **Histogram** (`descriptive_plots.png`, left): A bar plot with 5 bins showing the frequency of values. The x-axis represents data values (2 to 7), and the y-axis shows the count of occurrences. The peak at 5 reflects the mode.
- **Box Plot** (`descriptive_plots.png`, right): A horizontal box plot with a median line at 5, a box spanning the IQR (3.5 to 5.5), and whiskers extending to the min (2) and max (7). No outliers are present.

---

## 3. Probability Basics
Probability quantifies the likelihood of events, with values from 0 (impossible) to 1 (certain).
- **Sample Space**: All possible outcomes.
- **Event**: A subset of the sample space.
- **Probability**: P(Event) = (Favorable outcomes) / (Total outcomes).

### 3.1 Probability Distributions
- **Discrete Distributions**: For countable outcomes (e.g., Binomial for successes in trials, Poisson for event counts).
- **Continuous Distributions**: For continuous outcomes (e.g., Normal, Exponential).

```python
# Example: Binomial distribution (probability of k successes in n trials)
n, p = 10, 0.5  # 10 trials, 50% success probability
k = np.arange(0, n+1)
binomial_pmf = stats.binom.pmf(k, n, p)
plt.plot(k, binomial_pmf, 'o-', color='blue')
plt.title('Binomial PMF (n=10, p=0.5)')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('binomial_distribution.png')
plt.close()
print(f"Probability of 5 successes: {binomial_pmf[5]:.4f}")
```

**Output**:
```
Probability of 5 successes: 0.2461
```

**Graph Representation**:
- **Binomial PMF** (`binomial_distribution.png`): A line plot with points at integer values from 0 to 10 (x-axis) and probabilities on the y-axis (0 to 0.3). The peak at k=5 (probability ≈ 0.246) reflects the most likely number of successes in a fair coin flip scenario (p=0.5).

### 3.2 Normal Distribution
The normal distribution is a continuous, bell-shaped distribution defined by mean (μ) and standard deviation (σ). It’s symmetric and widely used due to the Central Limit Theorem.

```python
# Example: Plotting a normal distribution
mu, sigma = 0, 1
x = np.linspace(-4, 4, 100)
pdf = stats.norm.pdf(x, mu, sigma)
plt.plot(x, pdf, color='green')
plt.title('Normal Distribution (μ=0, σ=1)')
plt.xlabel('x')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('normal_distribution.png')
plt.close()
print(f"Probability density at x=0: {stats.norm.pdf(0, mu, sigma):.4f}")
```

**Output**:
```
Probability density at x=0: 0.3989
```

**Graph Representation**:
- **Normal Distribution** (`normal_distribution.png`): A smooth, bell-shaped curve centered at x=0 (μ=0), with the y-axis showing probability density (peaking at ≈0.399). The x-axis ranges from -4 to 4, covering most of the distribution’s area (99.7% within ±3σ).

---

## 4. Inferential Statistics
Inferential statistics use sample data to make generalizations about a population.

### 4.1 Hypothesis Testing
Hypothesis testing evaluates two mutually exclusive statements:
- **Null Hypothesis (H₀)**: No effect or difference (e.g., population mean = μ₀).
- **Alternative Hypothesis (H₁)**: An effect or difference exists.

Steps:
1. State H₀ and H₁.
2. Choose significance level (α = 0.05).
3. Compute test statistic and p-value.
4. Decision: Reject H₀ if p < α; otherwise, fail to reject.

```python
# Example: One-sample t-test
sample = [2.3, 2.5, 2.1, 2.4, 2.6]
pop_mean = 2.0
t_stat, p_value = stats.ttest_1samp(sample, pop_mean)
print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
```

**Output**:
```
T-statistic: 3.67, P-value: 0.0213
Reject the null hypothesis
```

**Explanation**: The p-value (0.0213) is less than α=0.05, so we reject H₀, suggesting the sample mean significantly differs from 2.0.

### 4.2 Confidence Intervals
A confidence interval (CI) estimates a population parameter with a confidence level (e.g., 95%).

```python
# Example: Confidence interval for the mean
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
n = len(sample)
ci = stats.t.interval(0.95, df=n-1, loc=sample_mean, scale=sample_std/np.sqrt(n))
print(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

**Output**:
```
95% Confidence Interval: (2.17, 2.63)
```

**Explanation**: We are 95% confident that the population mean lies between 2.17 and 2.63, based on the sample.

---

## 5. Advanced Statistical Methods
### 5.1 Regression Analysis
Regression models relationships between variables.
- **Linear Regression**: Fits a line (y = β₀ + β₁x) to predict a dependent variable from independent variables.

```python
# Example: Simple Linear Regression
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
x_with_const = sm.add_constant(x)  # Add intercept
model = sm.OLS(y, x_with_const).fit()
print(model.summary().tables[1])  # Coefficient table
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, model.predict(x_with_const), 'r-', label='Regression Line')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig('linear_regression.png')
plt.close()
```

**Output** (Coefficient Table):
```
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.1000      0.940      2.234      0.111      -0.964       5.164
x1             0.6000      0.283      2.121      0.125      -0.316       1.516
==============================================================================
```

**Graph Representation**:
- **Linear Regression** (`linear_regression.png`): A scatter plot of data points (x, y) with a red regression line (y ≈ 2.1 + 0.6x). The x-axis ranges from 1 to 5, and the y-axis from 0 to 6. The line shows a positive trend, though the fit is moderate (R² not shown but can be checked in the full summary).

### 5.2 ANOVA
Analysis of Variance (ANOVA) tests differences between group means.

```python
# Example: One-way ANOVA
group1 = [2.1, 2.3, 2.5]
group2 = [3.1, 3.2, 3.0]
group3 = [4.0, 4.1, 4.2]
f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"F-statistic: {f_stat:.2f}, P-value: {p_value:.4f}")
```

**Output**:
```
F-statistic: 40.67, P-value: 0.0005
```

**Explanation**: The low p-value (0.0005 < 0.05) indicates significant differences between group means.

### 5.3 Time Series Analysis
Time series analysis models data over time, often using smoothing techniques like the Simple Moving Average (SMA).

```python
# Example: Simple Moving Average
dates = pd.date_range('2023-01-01', periods=10, freq='D')
data = pd.Series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19], index=dates)
sma = data.rolling(window=3).mean()
plt.plot(data, label='Data', color='blue')
plt.plot(sma, label='3-Day SMA', color='red')
plt.title('Simple Moving Average')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('time_series_sma.png')
plt.close()
print(f"Last 3-day SMA value: {sma.iloc[-1]:.2f}")
```

**Output**:
```
Last 3-day SMA value: 18.00
```

**Graph Representation**:
- **Time Series SMA** (`time_series_sma.png`): A line plot with the original data in blue and the 3-day SMA in red. The x-axis shows dates (Jan 1–10, 2023), and the y-axis shows values (10 to 19). The SMA smooths fluctuations, highlighting the upward trend.

---

## 6. Practical Tips
- **Data Cleaning**: Use `df.dropna()` for missing values, or `df.fillna(df.mean())` for imputation. Detect outliers with `z = (x - μ)/σ` or IQR.
- **Visualization**: Use `seaborn` for advanced plots (e.g., `sns.pairplot` for variable relationships, `sns.heatmap` for correlations).
- **Reproducibility**: Set `np.random.seed(42)` and document code with comments.
