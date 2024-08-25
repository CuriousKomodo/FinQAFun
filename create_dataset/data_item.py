import dataclasses
from typing import Dict, Any, List

@dataclasses.dataclass
class DataItem:
    id: str
    pre_text: str
    post_text: str
    table: str
    question: str
    dialogue_break: List[str]
    step_list: List[str]
    answer_list: List[str]
    answer: str

