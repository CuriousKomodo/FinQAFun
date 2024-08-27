import os
import pandas as pd
import json

from data_item import DataItem

dir_path = os.path.dirname(os.path.realpath(__file__))

df = pd.read_json(f"{dir_path}/../data/train.json")
df.head()

num_type_1 = sum(df["qa_0"].isna())
num_type_2 = len(df)-num_type_1

type_1_df = df[df["qa_0"].isna()]

data_items = []
for idx, row in type_1_df[:100].iterrows():
    item = DataItem(
        id=row["id"],
        pre_text=row["pre_text"],
        post_text=row["post_text"],
        table=row["table"],
        question=row["qa"]["question"],
        dialogue_break=row["annotation"]["dialogue_break"],
        step_list=row["annotation"]["step_list"],
        answer_list=row["annotation"]["answer_list"],
        answer=row["qa"]["answer"],
    )
    data_items.append(item.__dict__)

    with open(f'{dir_path}/../data/train_data_items.json', 'w') as f:
        json.dump(data_items, f)

