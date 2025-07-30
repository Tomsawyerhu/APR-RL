import dataclasses
from enum import Enum


class DatasetType(Enum):
    HUMAN_EVAL = "human_eval"
    MBPP = "mbpp"
    CODE_FORCES = "code_forces"
    CODE_CONTESTS = "code_contests"


class ProblemType(Enum):
    CODE_GENERATION = 'code_generation'
    CODE_REPAIR = 'code_repair'


@dataclasses.dataclass
class CodeRepairProblem:
    dataset: str
    id: str
    question: str
    test_code: str
    test_inputs: list
    test_outputs: list
    entry_point: str
    ground_truth: str
    buggy_code: str
    difficulty: str = ''

    @classmethod
    def from_json(cls, json_dict: dict):
        data = json_dict

        item = cls(
            dataset=data["dataset"],
            id=data["id"],
            question=data['question'],
            test_code=data["test_code"],
            test_inputs=data["test_inputs"],
            test_outputs=data["test_outputs"],
            entry_point=data["entry_point"],
            ground_truth=data["ground_truth"],
            buggy_code=data["buggy_code"],
            difficulty=data["difficulty"]
        )
        return item


@dataclasses.dataclass
class CodeGenerationProblem:
    dataset: str
    id: str
    question: str
    test_code: str
    test_inputs: list
    test_outputs: list
    entry_point: str
    ground_truth: str
    difficulty: str = ''

    @classmethod
    def from_json(cls, json_dict: dict):
        data = json_dict

        item = cls(
            dataset=data["dataset"],
            id=data["id"],
            question=data['question'],
            test_code=data["test_code"],
            test_inputs=data["test_inputs"],
            test_outputs=data["test_outputs"],
            entry_point=data["entry_point"],
            ground_truth=data["ground_truth"],
            difficulty=data["difficulty"]
        )
        return item
