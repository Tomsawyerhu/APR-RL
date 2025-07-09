import gzip
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Dict
import os
import subprocess
from model import DatasetType, CodeRepairProblem
from exec_utils import check_correctness

test_dir = './tests'
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
logger = logging.getLogger(__name__)

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def split_problem_tests(problem: CodeRepairProblem):
    if problem.dataset == DatasetType.HUMAN_EVAL.value:
        pre_base_str, tests = problem.test_code.split("def check(candidate):\n")
        base_str = "def check(candidate):\n"
        split_tests = []
        # NOTE: assumes human-eval-specific logic for multiline asserts & for-loops
        # won't work properly if multiline assert nested within for-loop
        multiline_assert, parts = False, []
        for_loop, fl_parts = False, []
        for i in tests.split("\n"):
            if multiline_assert:
                parts.append(i)
                if i.lstrip().startswith("]"):
                    test = "\n".join(parts)
                    split_tests.append(pre_base_str + base_str + test)
                    multiline_assert = False
            elif i.lstrip().startswith("assert") and i.lstrip()[-1] == "[":
                multiline_assert = True
                parts = [i]
            elif for_loop:
                fl_parts.append(i)
                if i.lstrip().startswith("assert"):
                    test = "\n".join(fl_parts)
                    split_tests.append(pre_base_str + base_str + test)
                    for_loop, fl_parts = False, []
            elif (
                    (i.lstrip() == "")
                    or (i.lstrip().startswith("#"))
                    or (i.lstrip().startswith("print"))
            ):
                continue
            elif not (i.lstrip().startswith("assert")):
                fl_parts.append(i)
                if i.lstrip().startswith("for"):
                    for_loop = True
            # special logic for HumanEval/151
            elif problem.id == "HumanEval/151" and (
                    i.lstrip().startswith("assert candidate(lst)")
            ):
                fl_parts.append(i)
                test = "\n".join(fl_parts)
                split_tests.append(pre_base_str + base_str + test)
                fl_parts = []
            else:
                split_tests.append(pre_base_str + base_str + i)
        return split_tests
    elif problem.dataset == DatasetType.CODE_FORCES.value:
        split_tests = []
        for test_input, test_output in zip(problem.test_inputs, problem.test_outputs):
            split_tests.append({
                'test_input': test_input,
                'test_output': test_output
            })
        return split_tests
    else:
        raise Exception(f'unsupported dataset {problem.dataset} for test split')


def get_unique_id():
    unique_id = f"{int(time.time() * 1000)}_{random.randint(10000000, 99999999)}"
    return unique_id


def run_single_test(code, test_input, test_output, timeout=3):
    unique_file_name = get_unique_id() + ".py"
    file_path = os.path.join(test_dir, unique_file_name)

    with open(file_path, "w") as f:
        f.write(code)

    try:
        proc = subprocess.Popen(
            ["python", file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            stdout, stderr = proc.communicate(input=test_input, timeout=timeout)
            logger.info("stdout is %s, expect output is %s",stdout.strip(),test_output.strip())
            return_code = proc.returncode

            if stderr:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return {
                    "error": stderr.strip(),
                    "passed": False,
                }

            success = (stdout.strip() == test_output.strip())
            if os.path.exists(file_path):
                os.remove(file_path)
            return {
                "error": "",
                "passed": success,
            }

        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            if os.path.exists(file_path):
                os.remove(file_path)
            return {
                "error": "Execution timed out",
                "passed": False,
            }

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return {
            "error": str(e),
            "passed": False,
        }


def run_base_tests(problem: CodeRepairProblem, completion, timeout=1):
    if problem.dataset == DatasetType.HUMAN_EVAL.value:
        split_tests = split_problem_tests(problem)
        thread_problems = [CodeRepairProblem(test_code=test,
                                             id=problem.id,
                                             question=problem.question,
                                             dataset=problem.dataset,
                                             test_inputs=problem.test_inputs,
                                             test_outputs=problem.test_outputs,
                                             ground_truth=problem.ground_truth,
                                             entry_point=problem.entry_point,
                                             buggy_code=problem.buggy_code)
                           for test in split_tests]
        results = []
        with ThreadPoolExecutor() as executor:
            for result in executor.map(
                    lambda tp: check_correctness(tp, completion, timeout), thread_problems
            ):
                results.append(result["passed"])
        return {
            "task_id": problem.id,
            "pass_rate": sum(results) / len(results),
        }
    elif problem.dataset == DatasetType.MBPP.value:
        result = check_correctness(problem, completion, timeout)
        return {
            "task_id": problem.id,
            "passed": result['passed']
        }
    elif problem.dataset == DatasetType.CODE_FORCES.value:
        split_tests = split_problem_tests(problem)
        results = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(run_single_test, completion, t['test_input'], t['test_output'], timeout)
                for t in split_tests
            ]

            for future in as_completed(futures):
                result = future.result()
                if result['passed']:
                    results.append(1)
                else:
                    results.append(0)
        return {
            "task_id": problem.id,
            "pass_rate": sum(results) / len(results)
        }

    else:
        raise Exception(f'dataset {problem.dataset} is unsupported')


def run_extra_tests(problem: CodeRepairProblem, completion, extra_tests: list, timeout=3, check_on_gt=False):
    if len(extra_tests) == 0:
        raise Exception('empty test list')
    if problem.dataset == DatasetType.HUMAN_EVAL.value or problem.dataset == DatasetType.MBPP.value:
        results = []
        with ThreadPoolExecutor() as executor:
            for result in executor.map(
                    lambda extra_test: check_correctness(problem, completion, timeout, extra_assertion=extra_test,check_on_gt=check_on_gt),
                    extra_tests
            ):
                if isinstance(result,dict) and result.get('passed') is True:
                    results.append(1)
                else:
                    results.append(0)
        return {
            "task_id": problem.id,
            "pass_rate": sum(results) / len(results)
        }
    elif problem.dataset == DatasetType.CODE_FORCES.value:
        assert isinstance(extra_tests, list)
        results = []
        for extra_test in extra_tests:
            test_input = extra_test['test_input']
            test_output = extra_test['test_output']
            exec_result = run_single_test(completion, test_input, test_output, timeout)
            if exec_result['passed']:
                results.append(1)
            else:
                results.append(0)

        return {
            "task_id": problem.id,
            "pass_rate": sum(results) / len(results)
        }
    else:
        raise Exception(f'dataset {problem.dataset} unsupported')
