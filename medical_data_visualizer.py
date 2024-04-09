import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = np.where(bmi > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio',  'variable', 'value']).size().unstack(fill_value=0)
    df_cat.reset_index(inplace=True)
    df_cat = pd.melt(df_cat, id_vars=['cardio', 'variable'], value_vars=[0, 1], value_name='total')
    

    # Draw the Catplot with "sns.catplot"
    fig = sns.catplot(data=df_cat, x='variable', y='total', col='cardio', hue='value', kind='bar')

    # Get the figure
    fig = fig.figure


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(
        df['ap_lo'] <= df['ap_hi']) & ( # Diastolic blood pressure must be lower than Systolic blood pressure
        df['height'] >= df['height'].quantile(0.025)) & ( # Keeping rows where height is more or equal to the 2.5th percentile
        df['height'] <= df['height'].quantile(0.975)) & ( # Keeping rows where height is less or equal to the 97.5th percentile
        df['weight'] >= df['weight'].quantile(0.025)) & ( # Keeping rows where weight is more or equal to the 2.5th percentile
        df['weight'] <= df['weight'].quantile(0.975) # Keeping rows where weight is less or equal to the 97.5th percentile
    )]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, annot=True, fmt=".1f", cmap='coolwarm', center=0, mask=mask, square=True, ax=ax)


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
