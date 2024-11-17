import pickle
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import re
import pandas as pd

# Etape 1: essayer sans donner le format du log

config = TemplateMinerConfig()

# Step 1: Train Drain model on training log data
drain_parser = TemplateMiner(config=config)

log_pattern = re.compile(r"(?P<Date>\d+) (?P<Time>\d+) (?P<Pid>\d+) (?P<Level>\w+) (?P<Component>[^:]+): (?P<Content>.*)")

with open('../logs/HDFS_v3.log', 'r') as log_file:
    for line in log_file:
        # Remove any leading/trailing whitespace from the line
        line = line.strip()
        match = log_pattern.match(line)
        if match:
            log_content = match.group("Content")  # Extract the Content field
            result = drain_parser.add_log_message(log_content)

log_data = []
with open('../logs/HDFS_v3.log', 'r') as log_file:
    ct = 0
    for log in log_file:
        match = log_pattern.match(log)
        if match:
            log_content = match.group("Content")  # Extract the Content field
            date = match.group("Date")
            time = match.group("Time")
            info = match.group("Level")

            result = drain_parser.match(log_content)

            datef = '20'+date[0:2]+'-'+date[2:4]+'-'+date[4:6]
            timef = time[0:2]+':'+time[2:4]+':'+time[4:6]

            log_entry = {
                "Date": datef,
                "Time": timef,
                "Info": info,
                "Content": log_content,
                "Matched Template": result.get_template(),
                "Cluster ID": result.cluster_id
            }
            
            # Append the dictionary to the list
            log_data.append(log_entry)


df = pd.DataFrame(log_data)
df.to_csv('../data/parsed-hdfs-data.csv')

