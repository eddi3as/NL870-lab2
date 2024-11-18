import pickle
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import re
import pandas as pd

config = TemplateMinerConfig()

drain_parser = TemplateMiner(config=config)

log_pattern = re.compile(r"^(?P<log_file>[\w.-]+)\.(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})\s(?P<date>\d{4}-\d{2}-\d{2})\s(?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s(?P<process_id>\d+)\s(?P<log_level>\w+)\s(?P<source>[\w\.]+)\s(?P<content>.+)")

with open('../logs/OpenStack.log', 'r') as log_file:
    for line in log_file:
        line = line.strip()
        match = log_pattern.match(line)
        if match:
            log_content = match.group("content") 
            result = drain_parser.add_log_message(log_content)

log_data = []
with open('../logs/OpenStack.log', 'r') as log_file:
    for log in log_file:
        match = log_pattern.match(log)
        if match:
            log_content = match.group("content") 
            date = match.group("date")
            time = match.group("time")
            info = match.group("log_level")

            result = drain_parser.match(log_content)

            log_entry = {
                "Date": date,
                "Time": time,
                "Info": info,
                "Content": log_content,
                "Matched Template": result.get_template(),
                "Cluster ID": result.cluster_id
            }
            
            log_data.append(log_entry)


df = pd.DataFrame(log_data)
df.to_csv('../data/parsed-os-data.csv')

