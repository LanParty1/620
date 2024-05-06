import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
import numpy as np
import scipy as stats
import statsmodels.api as smf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf

def get_index_data(symbol, start_date, end_date):
    # Fetch historical market data
    data = yf.download(symbol, start=start_date, end=end_date)['Close']
    return data

def calculate_rate_of_return(data):
    # Calculate the rate of return
    return (data[-1] - data[0]) / data[0] * 100

def calculate_standard_deviation(data):
    # Calculate the standard deviation
    return np.std(data)
def perform_multivariate_regression(X, y):
    # Add additional features to X (multivariate regression)
    X = np.column_stack((X, X**2))  # Example: Adding squared time as an additional feature
    model = lr().fit(X, y)
    return model

def perform_nonlinear_regression(X, y, degree):
    # Fit a polynomial regression model
    coeffs = np.polyfit(X.flatten(), y, degree)
    return coeffs

def perform_regression_analysis(X, y):
    # Perform linear regression and calculate R-squared
    X_1d = X.flatten()  # Convert X to a 1D array
    slope, intercept = np.polyfit(X_1d, y, 1)
    residuals = y - (slope * X_1d + intercept)
    r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
    return slope, intercept, r_squared


if __name__ == "__main__":
    #Time period
    start_date = '2000-01-01'
    end_date = '2023-12-31'

    #Data for Nasdaq and S&P 500
    nasdaq_data = get_index_data('^IXIC', start_date, end_date)
    sp_data = get_index_data('^GSPC', start_date, end_date)

    #Lengths of data arrays
    if len(nasdaq_data) != len(sp_data):
        raise ValueError("Lengths of data arrays are not the same.")

    #RoR
    nasdaq_rate_of_return = calculate_rate_of_return(nasdaq_data)
    sp_rate_of_return = calculate_rate_of_return(sp_data)

    #SD
    nasdaq_std_dev = calculate_standard_deviation(nasdaq_data)
    sp_std_dev = calculate_standard_deviation(sp_data)

    #Linear regression and R-squared for Nasdaq
    X_nasdaq = np.arange(len(nasdaq_data)).reshape(-1, 1)
    nasdaq_slope, nasdaq_intercept, nasdaq_r_squared = perform_regression_analysis(X_nasdaq, nasdaq_data.values)

    #Linear regression and R-squared for S&P 500
    X_sp = np.arange(len(sp_data)).reshape(-1, 1)
    sp_slope, sp_intercept, sp_r_squared = perform_regression_analysis(X_sp, sp_data.values)

    #Hypothesis testing
    t_stat, p_value = ttest_ind(nasdaq_data, sp_data, equal_var=False)

    #Correlation coefficient
    correlation_coefficient = np.corrcoef(nasdaq_data, sp_data)[0, 1]
    
    #Additional features for multivariate regression
    X_nasdaq_multivariate = np.column_stack((X_nasdaq, X_nasdaq**2))  # Example: Adding squared time as an additional feature
    X_sp_multivariate = np.column_stack((X_sp, X_sp**2))  # Example: Adding squared time as an additional feature

    #Perform multivariate regression
    nasdaq_multivariate_model = perform_multivariate_regression(X_nasdaq_multivariate, nasdaq_data.values)
    sp_multivariate_model = perform_multivariate_regression(X_sp_multivariate, sp_data.values)

    #Perform nonlinear regression
    degree = 3  # Example: Using a cubic polynomial
    nasdaq_nonlinear_coeffs = perform_nonlinear_regression(X_nasdaq, nasdaq_data.values, degree)
    sp_nonlinear_coeffs = perform_nonlinear_regression(X_sp, sp_data.values, degree)

    #Plotting nonlinear regression
    nasdaq_nonlinear_reg_line = np.polyval(nasdaq_nonlinear_coeffs, X_nasdaq.flatten())
    sp_nonlinear_reg_line = np.polyval(sp_nonlinear_coeffs, X_sp.flatten())

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    #Nasdaq Nonlinear Regression
    axs[0].scatter(nasdaq_data.index, nasdaq_data.values, alpha=0.5, color='blue', label='Nasdaq')
    axs[0].plot(nasdaq_data.index, nasdaq_nonlinear_reg_line, color='red', linestyle='--', label='Nasdaq Nonlinear Regression Line')
    axs[0].set_title('Nasdaq Prices with Nonlinear Regression')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    #S&P 500 Nonlinear Regression
    axs[1].scatter(sp_data.index, sp_data.values, alpha=0.5, color='red', label='S&P 500')
    axs[1].plot(sp_data.index, sp_nonlinear_reg_line, color='blue', linestyle='--', label='S&P 500 Nonlinear Regression Line')
    axs[1].set_title('S&P 500 Prices with Nonlinear Regression')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    print("Nasdaq:")
    print("Rate of return: {:.2f}%".format(nasdaq_rate_of_return))
    print("Standard deviation: {:.2f}".format(nasdaq_std_dev))
    print("Trend (slope): {:.6f}".format(nasdaq_slope))
    print("R-squared: {:.4f}".format(nasdaq_r_squared))
    print("Intercept of regression line: {:.6f}".format(nasdaq_intercept))
    print("\nS&P 500:")
    print("Rate of return: {:.2f}%".format(sp_rate_of_return))
    print("Standard deviation: {:.2f}".format(sp_std_dev))
    print("Trend (slope): {:.6f}".format(sp_slope))
    print("R-squared: {:.4f}".format(sp_r_squared))
    print("Intercept of regression line: {:.6f}".format(sp_intercept))

    #Hypothesis testing results
    print("\nHypothesis Testing:")
    print("T-statistic:", t_stat)
    print("P-value:", p_value)
    if p_value < 0.05:
        print("Null hypothesis rejected: There is a significant difference in means.")
    else:
        print("Failed to reject null hypothesis: There is no significant difference in means.")
        
    #Compare stability and rate of return
    if nasdaq_rate_of_return > sp_rate_of_return:
        print("\nThe Nasdaq has generated a higher rate of return compared to the S&P 500.")
    else:
        print("\nThe S&P 500 has generated a higher rate of return compared to the Nasdaq.")

    if nasdaq_std_dev < sp_std_dev:
        print("The Nasdaq has proven to be more stable compared to the S&P 500.")
    else:
        print("The S&P 500 has proven to be more stable compared to the Nasdaq.")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    #Nasdaq prices
    axs[0, 0].plot(nasdaq_data, color='blue')
    axs[0, 0].set_title('Nasdaq Prices')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Price')

    #S&P 500 prices
    axs[0, 1].plot(sp_data, color='red')
    axs[0, 1].set_title('S&P 500 Prices')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Price')

    #Relationship Nasdaq S&P 500
    nasdaq_reg_line = nasdaq_slope * X_nasdaq.flatten() + nasdaq_intercept
    sp_reg_line = sp_slope * X_sp.flatten() + sp_intercept
    nasdaq_sp_diff = sp_data.values - nasdaq_data.values

    colors = np.where(nasdaq_sp_diff > 0, 'green', 'red')

    #Plot correlation indication
    axs[1, 0].scatter(nasdaq_data.values, sp_data.values, alpha=0.5, c=colors)
    axs[1, 0].plot(nasdaq_data.values, nasdaq_reg_line, color='blue', linestyle='-', linewidth=2, label='Nasdaq Regression Line')
    axs[1, 0].plot(sp_data.values, sp_reg_line, color='orange', linestyle='-', linewidth=2, label='S&P 500 Regression Line')
    axs[1, 0].set_title('Relationship between Nasdaq and S&P 500')
    axs[1, 0].set_xlabel('Nasdaq')
    axs[1, 0].set_ylabel('S&P 500')

    #Adjust
    max_val = max(np.max(nasdaq_data.values), np.max(sp_data.values))
    min_val = min(np.min(nasdaq_data.values), np.min(sp_data.values))
    axs[1, 0].set_xlim(min_val, max_val)
    axs[1, 0].set_ylim(min_val, max_val)
    
    axs[1, 0].legend()
    
    #Nasdaq Prices against Dates
    axs[1, 1].scatter(nasdaq_data.index, nasdaq_data.values, alpha=0.5, color='blue', label='Nasdaq')
    #S&P 500 Prices against Dates
    axs[1, 1].scatter(sp_data.index, sp_data.values, alpha=0.5, color='red', label='S&P 500')
   
    nasdaq_slope, nasdaq_intercept, _ = perform_regression_analysis(X_nasdaq, nasdaq_data.values)
    nasdaq_reg_line = nasdaq_slope * np.arange(len(nasdaq_data)) + nasdaq_intercept
    axs[1, 1].plot(nasdaq_data.index, nasdaq_reg_line, color='blue', linestyle='--', label='Nasdaq Regression Line')

    sp_slope, sp_intercept, _ = perform_regression_analysis(X_sp, sp_data.values)
    sp_reg_line = sp_slope * np.arange(len(sp_data)) + sp_intercept
    axs[1, 1].plot(sp_data.index, sp_reg_line, color='red', linestyle='--', label='S&P 500 Regression Line')

    axs[1, 1].set_title('Nasdaq and S&P 500 Prices')
    axs[1, 1].set_xlabel('Date')
    axs[1, 1].set_ylabel('Price')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

#Plot line graph utilizing statistical analysis: regression, SD, and mean
fig, axs = plt.subplots(figsize=(10, 6))
axs.plot(nasdaq_data.index, nasdaq_data.values, label='Nasdaq Prices')

#Regression Nasdaq
nasdaq_reg_line = nasdaq_slope * X_nasdaq.flatten() + nasdaq_intercept
axs.plot(nasdaq_data.index, nasdaq_reg_line, color='red', linestyle='--', label='Nasdaq Regression Line')

# Highlight areas within one standard deviation from the regression line
nasdaq_std = np.std(nasdaq_data.values)
axs.fill_between(nasdaq_data.index, nasdaq_reg_line - nasdaq_std, nasdaq_reg_line + nasdaq_std, color='blue', alpha=0.3, label='1 SD')

axs.plot(sp_data.index, sp_data.values, label='S&P 500 Prices')

#Regression S&P 500
sp_reg_line = sp_slope * X_sp.flatten() + sp_intercept
axs.plot(sp_data.index, sp_reg_line, color='green', linestyle='--', label='S&P 500 Regression Line')

# Highlight areas within one standard deviation from the regression line
sp_std = np.std(sp_data.values)
axs.fill_between(sp_data.index, sp_reg_line - sp_std, sp_reg_line + sp_std, color='orange', alpha=0.3, label='1 SD')

axs.legend()
axs.set_xlabel('Date')
axs.set_ylabel('Price')
axs.set_title('Nasdaq and S&P 500 Prices with Regression Lines and Standard Deviation')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
