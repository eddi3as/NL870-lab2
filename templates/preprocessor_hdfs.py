import pandas as pd

def split_and_aggregate_by_cluster(df, time_interval='30T', error_threshold=5, anomaly_clusters=None):

    df.set_index('Datetime', inplace=True)

    cluster_counts = df.groupby([pd.Grouper(freq=time_interval), 'Cluster ID']).size().unstack(fill_value=0)
    cluster_counts['Anomaly'] = '0'

    for index, row in cluster_counts.iterrows():
        if anomaly_clusters:
            if row[anomaly_clusters].sum() >= error_threshold:
                cluster_counts.at[index, 'Anomaly'] = '1'
           
    if anomaly_clusters:
        cluster_counts.drop(columns=anomaly_clusters, inplace=True)

    return cluster_counts


time_interval = '30T'
anomaly_clusters = [9]  # Cluster IDs ou se trouve les anomalies
error_threshold = 5


df = pd.read_csv('../data/parsed-hdfs-data.csv')

datetimeft = df['Date'] + ' ' + df['Time']
df['Datetime'] = pd.to_datetime(datetimeft)

df = df.sort_values(by='Datetime')

result_df = split_and_aggregate_by_cluster(df, time_interval, error_threshold, anomaly_clusters)

result_df.reset_index(inplace=True)

print(result_df)

result_df.to_csv('../data/parsed-hdfs-aggregated.csv', index=False)