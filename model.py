import dataclasses
from enum import Enum


class DatasetType(Enum):
    HUMAN_EVAL = "human_eval"
    MBPP = "mbpp"
    CODE_FORCES = "code_forces"


@dataclasses.dataclass
class CodeRepairProblem:
    dataset: str
    id: str
    question:str
    test_code: str
    test_inputs: list
    test_outputs: list
    entry_point: str
    ground_truth: str
    buggy_code: str

    @classmethod
    def from_json(cls, json_dict: dict):
        data = json_dict

        return cls(
            dataset=data["dataset"],
            id=data["id"],
            question=data['question'],
            test_code=data["test_code"],
            test_inputs=data["test_inputs"],
            test_outputs=data["test_outputs"],
            entry_point=data["entry_point"],
            ground_truth=data["ground_truth"],
            buggy_code=data["buggy_code"]
        )
