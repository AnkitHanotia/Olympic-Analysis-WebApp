import numpy as np
import pandas as pd


def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == year) & (medal_df['region'] == country)]

    if flag == 1:
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold',
                                                                                      ascending=False).reset_index()

    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    x['Gold'] = x['Gold'].astype('int')
    x['Silver'] = x['Silver'].astype('int')
    x['Bronze'] = x['Bronze'].astype('int')
    x['total'] = x['total'].astype('int')

    return x


def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years, country


def data_over_time(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Analyzes the number of unique entries in a specific column over different years.

    :param df: The input DataFrame containing the data.
    :param col: The column for which to count unique entries.
    :return: A DataFrame showing the count of unique entries over the years.
    """
    # Drop duplicates based on Year and the specified column
    unique_entries = df.drop_duplicates(['Year', col])

    # Group by 'Year' and count the number of unique entries in the specified column
    nations_over_time = unique_entries.groupby('Year').size().reset_index(name=col)

    # Rename columns for clarity
    nations_over_time.rename(columns={'Year': 'Edition'}, inplace=True)

    # Sort by 'Edition' to ensure chronological order
    nations_over_time.sort_values('Edition', inplace=True)

    # Reset index for the final DataFrame
    nations_over_time.reset_index(drop=True, inplace=True)

    return nations_over_time


def most_successful(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """
    Analyzes the top 15 most successful athletes based on the number of medals won in a specific sport or overall.

    :param df: The input DataFrame containing the data.
    :param sport: The sport to filter by. Use 'Overall' to include all sports.
    :return: A DataFrame with the top 15 athletes, including their medals count, sport, and region.
    """
    # Drop rows with missing medals
    temp_df = df.dropna(subset=['Medal'])

    # Filter by sport if specified
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    # Count medals by athlete and select the top 15
    top15 = temp_df['Name'].value_counts().reset_index()
    top15.columns = ['Name', 'Medals']  # Rename columns for clarity

    # Merge to get additional details
    top15_df = top15.merge(df[['Name', 'Sport', 'region']], on='Name', how='left')

    # Remove duplicates
    top15_df = top15_df[['Name', 'Medals', 'Sport', 'region']].drop_duplicates()

    # Sort by number of medals
    top15_df = top15_df.sort_values(by='Medals', ascending=False).head(15)

    return top15_df


def yearwise_medal_tally(df, country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df


def country_event_heatmap(df, country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]

    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt


def most_successful_countrywise(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Analyzes the top 10 most successful athletes from a specific country based on the number of medals won.

    :param df: The input DataFrame containing the data.
    :param country: The country for which to analyze the top athletes.
    :return: A DataFrame with the top 10 athletes from the specified country, including their medals count and sport.
    """
    # Drop rows with missing medals
    temp_df = df.dropna(subset=['Medal'])

    # Filter for the specified country
    temp_df = temp_df[temp_df['region'] == country]

    # Count medals by athlete
    top10 = temp_df['Name'].value_counts().reset_index()
    top10.columns = ['Name', 'Medals']  # Rename columns for clarity

    # Merge to get additional details
    top10_df = top10.merge(temp_df[['Name', 'Sport', 'region']], on='Name', how='left')

    # Remove duplicates
    top10_df = top10_df[['Name', 'Medals', 'Sport', 'region']].drop_duplicates()

    return top10_df


def weight_v_height(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df


def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final
