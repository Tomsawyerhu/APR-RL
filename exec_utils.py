import logging
import random
import subprocess
import time
import uuid
from typing import Optional, Callable, Dict
import contextlib
import io
import os
import multiprocessing
import signal
import tempfile

from model import DatasetType, CodeRepairProblem

logger = logging.getLogger(__name__)


def get_unique_id():
    return str(uuid.uuid4().hex)  # 生成 32 位十六进制字符串


def unsafe_execute(problem: CodeRepairProblem, completion, result, timeout, extra_assertion=None):
    # import traceback
    # import linecache
    # import unittest.mock
    # logger.info('run unsafe_exec')
    # with open('./log.text', 'a') as f:
    #     f.write('run unsafe_exec\n\n')
    #
    # orig_getline = linecache.getline

    # def new_getline(filename, lineno, *args, **kwargs):
    #     if filename == "<string>":
    #         return check_program.splitlines()[lineno - 1]
    #     return orig_getline(filename, lineno, *args, **kwargs)

    # with create_tempdir():
    #
    #     # These system calls are needed when cleaning up tempdir.
    #     import os
    #     import shutil
    #     rmtree = shutil.rmtree
    #     rmdir = os.rmdir
    #     chdir = os.chdir
    #
    #     # Construct the check program and run it.
    #     if problem.dataset == DatasetType.HUMAN_EVAL.value:
    #         if extra_assertion:
    #             check_program = (
    #                     completion + "\n" + extra_assertion
    #             )
    #         else:
    #             check_program = (
    #                     completion + "\n" +
    #                     problem.test_code + "\n" +
    #                     f"check({problem.entry_point})"
    #             )
    #     elif problem.dataset == DatasetType.MBPP.value:
    #         if extra_assertion:
    #             check_program = (
    #                     completion + "\n" + extra_assertion
    #             )
    #         else:
    #             check_program = (
    #                     completion + "\n" + problem.test_code
    #             )
    #     else:
    #         raise Exception(f'dataset {problem.dataset} unsupported')
    #
    #     logger.info("check program is: %s", check_program)
    #     # with open('./log.text','a') as f:
    #     #     f.write(check_program+'\n\n')
    #
    #     with unittest.mock.patch("linecache.getline", new_getline):
    #         try:
    #             exec_globals = {}
    #             with swallow_io():
    #                 with time_limit(timeout):
    #                     exec(check_program, exec_globals)
    #             result.append("passed")
    #             logger.info("check result is passed")
    #         except TimeoutException:
    #             result.append("timed out")
    #             logger.info("check result is timed out")
    #         except BaseException as e:
    #             tb = traceback.TracebackException.from_exception(e)
    #             formatted_tb = ''.join(tb.format())
    #             result.append(f"failed: {e}, traceback: {formatted_tb}")
    #             logger.info("check result is failed")
    #
    #     # Needed for cleaning up.
    #     shutil.rmtree = rmtree
    #     os.rmdir = rmdir
    #     os.chdir = chdir
    # Construct the check program and run it.
    if problem.dataset == DatasetType.HUMAN_EVAL.value:
        if extra_assertion:
            check_program = (
                    completion + "\n" + extra_assertion
            )
        else:
            check_program = (
                    completion + "\n" +
                    problem.test_code + "\n" +
                    f"check({problem.entry_point})"
            )
    elif problem.dataset == DatasetType.MBPP.value:
        if extra_assertion:
            check_program = (
                    completion + "\n" + extra_assertion
            )
        else:
            check_program = (
                    completion + "\n" + problem.test_code
            )
    else:
        raise Exception(f'dataset {problem.dataset} unsupported')

    logger.info("check program is: %s", check_program)
    # with open('./log.text','a') as f:
    #     f.write(check_program+'\n\n')

    script_path = f'./tests/{get_unique_id()}.py'
    with open(script_path, 'w') as f:
        f.write(check_program)

    try:
        exec_result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if exec_result.returncode == 0:
            result.append("passed")
        else:
            result.append(f"failed: {exec_result.stderr}")
    except subprocess.TimeoutExpired:
        result.append("timed out")
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def check_correctness(problem: CodeRepairProblem, completion: str, timeout: float,
                      completion_id: Optional[int] = None, extra_assertion=None,check_on_gt=False) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    # manager = multiprocessing.Manager()
    # result = manager.list()
    #
    # # p = multiprocessing.Process(target=unsafe_execute)
    # p = multiprocessing.Process(
    #     target=unsafe_execute,
    #     args=(
    #         problem,
    #         completion,
    #         result,
    #         timeout,
    #         extra_assertion
    #     ),
    # )
    # p.start()
    # p.join(timeout=timeout + 1)
    # if p.is_alive():
    #     p.kill()
    if extra_assertion is not None and not isinstance(extra_assertion, str):
        return dict(
            task_id=problem.id,
            test=problem.test_code if extra_assertion is None else extra_assertion,
            passed=False,
            result='',
            completion_id=completion_id,
        )
    result = []
    unsafe_execute(problem, completion, result, timeout, extra_assertion)

    if not result:
        result.append("timed out")
    if extra_assertion is None:
        logger.info('exec test result: %s', result[0])
    else:
        if check_on_gt:
            logger.info('evaluate generated test on gt: %s', result[0])
        else:
            logger.info('evaluate generated test on bug: %s', result[0])


    return dict(
        task_id=problem.id,
        test=problem.test_code if extra_assertion is None else extra_assertion,
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)
