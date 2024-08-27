import json
import os

from create_dataset.data_item import DataItem
from pipeline.evaluation import evaluate
from pipeline.run_pipeline import execute_inference


if __name__ == '__main__':
    dir_path = os.getenv("DATA_DIR")
    data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))

    id_to_infer = "Single_JKHY/2009/page_28.pdf-3"
    data_item = [item for item in data_items if item["id"] == id_to_infer][0]
    data_item_obj = DataItem(**data_item)
    output = execute_inference(data_item_obj)

    metrics = evaluate(output=output, data_item=data_item)

    print(f"Final answer: {output['final_output']} \n")
    print(f"Metrics: {metrics} \n")

