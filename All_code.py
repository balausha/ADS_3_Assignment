import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# plt.style.use('dark_background')


# Load the data
df = pd.read_csv('./Data.csv', on_bad_lines='skip')

# Define the countries and indicators to plot
countries = ['United States', 'China', 'India', 'Brazil']
indicators = ['GDP (current US$)', 'Population, total', 'CO2 emissions (metric tons per capita)', 'Life expectancy at birth, total (years)']

# Set the figure size and style
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')

# Plot the data for each country and indicator
for country in countries:
    for indicator in indicators:
        subset = df[(df['Country Name'] == country) & (df['Indicator Name'] == indicator)]
        if not subset.empty:
            # Get the values for the last 10 years
            values = subset.iloc[0, -10:].values
            years = subset.columns[-10:].values

            # Plot the data
            plt.plot(years, values, label=indicator)

    # Set the x-axis label and limit
    plt.xlabel('Year')
    plt.xticks(rotation=45, ha='right')
    plt.xlim(years[0], years[-1])

    # Set the y-axis label
    plt.ylabel('Value')

    # Set the title and legend
    plt.title(f'{country}')
    plt.legend(loc='upper left')
    plt.show()



# Filter the data for the year 1960 and the 'Urban population (% of total population)' indicator
filtered_data = df.loc[(df['Indicator Name'] == 'Urban population (% of total population)') & (df['1960'].notnull())]

# Sort the data by the indicator value and take only the top ten rows
sorted_data = filtered_data.sort_values('1960', ascending=False)[:10]

# Create a bar chart
plt.bar(sorted_data['Country Name'], sorted_data['1960'])

# Set the chart title and axis labels
plt.title('Urban population (% of total population) in 1960')
plt.xlabel('Country')
plt.ylabel('Urban population (% of total population)')

# Rotate the x-axis labels to avoid overlapping
plt.xticks(rotation=90)

# Display the chart
plt.show()




indicators = ['GDP per capita (constant 2010 US$)', 'CO2 emissions (metric tons per capita)']
data = df.loc[df['Indicator Name'].isin(indicators)]
data = data.pivot_table(index=['Country Name', 'Country Code'], columns=['Indicator Name'], values='2019')
data.dropna(inplace=True)

# Normalize the data
data_norm = (data - data.mean()) / data.std()

# Cluster the countries
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(data_norm)

# Visualize the results
fig, ax = plt.subplots()
scatter = ax.scatter(data_norm.iloc[:, 0], data_norm.iloc[:, 0], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.xlabel('GDP per capita (normalized)')
plt.ylabel('CO2 emissions per capita (normalized)')
plt.show()



indicators = ['GDP per capita (constant 2010 US$)', 'CO2 emissions (metric tons per capita)']
data = df.loc[df['Indicator Name'].isin(indicators)]
data = data.pivot_table(index=['Country Name', 'Country Code'], columns=['Indicator Name'], values='1990')
data.dropna(inplace=True)

# Normalize the data
data_norm = (data - data.mean()) / data.std()

# Cluster the countries
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(data_norm)

# Visualize the results
fig, ax = plt.subplots()
scatter = ax.scatter(data_norm.iloc[:, 0], data_norm.iloc[:, 0], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.xlabel('GDP per capita (normalized)')
plt.ylabel('CO2 emissions per capita (normalized)')
plt.show()




""" Tools to support clustering: correlation heatmap, normaliser and scale 
(cluster centres) back to original scale, check for mismatching entries """


def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of 
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
        
    The function does not have a plt.show() at the end so that the user 
    can savethe figure.
    """

    import matplotlib.pyplot as plt  # ensure pyplot imported

    corr = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm', location="bottom")
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.colorbar()
    # no plt.show() at the end
    
    
def scaler(df):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr


def get_diff_entries(df1, df2, column):
    """ Compares the values of column in df1 and the column with the same 
    name in df2. A list of mismatching entries is returned. The list will be
    empty if all entries match. """

    import pandas as pd  # to be sure

    # merge dataframes keeping all rows
    df_out = pd.merge(df1, df2, on=column, how="outer")
    print("total entries", len(df_out))
    # merge keeping only rows in common
    df_in = pd.merge(df1, df2, on=column, how="inner")
    print("entries in common", len(df_in))
    df_in["exists"] = "Y"

    # merge again
    df_merge = pd.merge(df_out, df_in, on=column, how="outer")

    # extract columns without "Y" in exists
    df_diff = df_merge[(df_merge["exists"] != "Y")]
    diff_list = df_diff[column].to_list()

    return diff_list


""" Module errors. It provides the function err_ranges which calculates upper
and lower limits of the confidence interval. """

import numpy as np


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# Load the data
df = pd.read_csv('./Data.csv', on_bad_lines='skip')


indicators = ['GDP per capita (constant 2010 US$)', 'CO2 emissions (metric tons per capita)']
data = df.loc[df['Indicator Name'].isin(indicators)]
data = data.pivot_table(index=['Country Name', 'Country Code'], columns=['Indicator Name'], values='2019')
data.fillna(0, inplace=True)

def polynomial(x, a, b, c):
    return a*x**2 + b*x + c

# Get the x and y values
x = np.arange(len(data))
y = data.iloc[:, 0].values

# Define the range of data to be plotted
start_index = 0  # Specify the starting index
end_index = 20  # Specify the ending index

# Slice the data within the specified range
x_plot = x[start_index:end_index]
y_plot = y[start_index:end_index]

# Fit the data to the model
popt, pcov = curve_fit(polynomial, x_plot, y_plot)

# Generate x values for the plotted data
x_plot_pred = np.arange(start_index, end_index)

# Predict the y values for the plotted data
y_plot_pred = polynomial(x_plot_pred, *popt)

# Generate x values for the next 10 years
x_pred = np.arange(end_index, end_index + 10)

# Predict the y values for the next 10 years
y_pred = polynomial(x_pred, *popt)

# Calculate the confidence intervals
sigma = np.sqrt(np.diag(pcov))
lower, upper = err_ranges(x_pred, polynomial, popt, sigma)

# Rescale data and predictions
df_norm, df_min, df_max = scaler(data)
y_norm = (y_plot - df_min.iloc[0]) / (df_max.iloc[0] - df_min.iloc[0])
y_pred_norm = (y_pred - df_min.iloc[0]) / (df_max.iloc[0] - df_min.iloc[0])

# Fit the normalised data to the model
popt_norm, pcov_norm = curve_fit(polynomial, x_plot, y_norm)

# Predict the normalised y values for the next 10 years
y_pred_norm = polynomial(x_pred, *popt_norm)

# Backscale the predictions to the original scale
y_pred_backscaled = backscale(y_pred_norm, df_min.iloc[1:], df_max.iloc[1:])

# Calculate the confidence intervals
sigma_norm = np.sqrt(np.diag(pcov_norm))
lower_norm, upper_norm = err_ranges(x_pred, polynomial, popt_norm, sigma_norm)
lower_backscaled = backscale(lower_norm, df_min.iloc[1:], df_max.iloc[1:])
upper_backscaled = backscale(upper_norm, df_min.iloc[1:], df_max.iloc[1:])

# Plot the data and the fitted curve
plt.plot(df_norm.index.get_level_values(0)[start_index:end_index], y_norm, label='Data')

plt.xticks(rotation='vertical')
plt.legend()
plt.show()

