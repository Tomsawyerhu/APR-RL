import re
from collections import defaultdict

from model import DatasetType

# 日志文件路径（假设你的日志已经保存为一个文本文件）
LOG_FILE = "/Users/tom/PycharmProjects/aileetcode/logs/rl/qwen/4b/logfile.txt"


fix_only_mode = False
# 初始化统计字典：dataset -> 指标计数器
stats = defaultdict(lambda: {
    'total_samples': 0,
    'fixed_code_pass_count': 0,
    'testcase_validity_sum': 0.0,
    'non_zero_coverage_count': 0,
    'pass_pass': 0,
    'pass_fail': 0,
    'fail_fail': 0,
    'fail_pass': 0,
})

# 正则表达式匹配不同 dataset 的固定代码奖励和测试用例奖励
pattern_fixed = r"dataset = (\w+), fixed_code_pass_all_test_reward is \[(.*?)\]"
pattern_testcase = r"dataset = (\w+), testcase_pass_groundtruth_and_kill_bug_reward is \[(.*?)\]"

with open(LOG_FILE, 'r') as f:
    log_lines = f.read()

# 提取所有 dataset 名称及对应的奖励值
fixed_rewards_dict = defaultdict(list)
testcase_rewards_dict = defaultdict(list)

for match in re.finditer(pattern_fixed, log_lines):
    dataset = match.group(1)
    rewards_str = match.group(2)
    rewards = list(map(float, rewards_str.split(', ')))
    fixed_rewards_dict[dataset].append(rewards)

if not fix_only_mode:
    for match in re.finditer(pattern_testcase, log_lines):
        dataset = match.group(1)
        rewards_str = match.group(2)
        rewards = list(map(float, rewards_str.split(', ')))
        testcase_rewards_dict[dataset].append(rewards)

# 设置 pass_k 值
pass_k = 3

# 确保每个 dataset 下两个列表长度一致
# for dataset in fixed_rewards_dict:
#     assert len(fixed_rewards_dict[dataset]) == len(testcase_rewards_dict.get(dataset, [])), \
#         f"Dataset {dataset} has inconsistent number of log entries."

pass_pass, pass_fail, fail_pass, fail_fail, sum_number = 0, 0, 0, 0,0
# 统计每个 dataset 的结果
for dataset in fixed_rewards_dict:
    for i in range(len(fixed_rewards_dict[dataset])):
        stats[dataset]['total_samples'] += 1
        sum_number+=1

        fixed_rewards = fixed_rewards_dict[dataset][i]
        if not fix_only_mode:
            testcase_rewards = testcase_rewards_dict[dataset][i]

        # 指标 1: 修复代码通过率（pass rate == 1.0）
        if any(x == 1.0 for x in fixed_rewards[:pass_k]):
            stats[dataset]['fixed_code_pass_count'] += 1
        if not fix_only_mode:
            # 指标 2: 测试用例有效性通过率（平均分）
            stats[dataset]['testcase_validity_sum'] += sum(testcase_rewards[:pass_k]) / pass_k

            # 指标 3: 非零覆盖率比率（至少有一个 > 0）
            if any(x != 0 for x in testcase_rewards[:pass_k]):
                stats[dataset]['non_zero_coverage_count'] += 1

            # 指标4-7 测试用例有效和修复成功的关系
            for x, y in zip(testcase_rewards[:pass_k], fixed_rewards[:pass_k]):
                if x > 0 and y > 0:
                    stats[dataset]['pass_pass'] += 1
                    pass_pass += 1
                elif x > 0 and y == 0:
                    stats[dataset]['pass_fail'] += 1
                    pass_fail += 1
                elif x == 0 and y > 0:
                    stats[dataset]['fail_pass'] += 1
                    fail_pass += 1
                elif x == 0 and y == 0:
                    stats[dataset]['fail_fail'] += 1
                    fail_fail += 1

# 输出结果
print("=== Summary ===")
dataset_order = [DatasetType.HUMAN_EVAL.value, DatasetType.MBPP.value, DatasetType.CODE_FORCES.value,
                 DatasetType.CODE_CONTESTS.value]

tex_str = ''

for dataset in dataset_order:
    data = stats[dataset]
    total = data['total_samples']
    if total == 0:
        continue  # 跳过没有数据的 dataset

    fixed_rate = data['fixed_code_pass_count'] / total
    if not fix_only_mode:
        validity_avg = data['testcase_validity_sum'] / total
        non_zero_ratio = data['non_zero_coverage_count'] / total

    print(f"\n--- Dataset: {dataset} ---")
    print(f"Total samples: {total}")
    print(
        f"1. Fixed Code Pass Rate @ {pass_k}: {data['fixed_code_pass_count']} / {total} = {100 * fixed_rate:.2f}")
    if not fix_only_mode:
        print(
            f"2. Testcase Validity Avg @ {pass_k}: {100 * validity_avg:.2f}")
        print(
            f"3. Non-Zero Coverage Ratio @ {pass_k}: {data['non_zero_coverage_count']} / {total} = {100 * non_zero_ratio:.2f}")
        print(
            f"4. Valid Test Found and Fix Success @ {pass_k}: = {100 * data['pass_pass'] / (data['pass_pass'] + data['pass_fail'] + data['fail_pass'] + data['fail_fail']):.2f}")
        print(
            f"5. Valid Test Found and Fix Fail @ {pass_k}: = {100 * data['pass_fail'] / (data['pass_pass'] + data['pass_fail'] + data['fail_pass'] + data['fail_fail']):.2f}")
        print(
            f"6. Valid Test not Found and Fix Success @ {pass_k}: = {100 * data['fail_pass'] / (data['pass_pass'] + data['pass_fail'] + data['fail_pass'] + data['fail_fail']):.2f}")
        print(
            f"7. Valid Test not Found and Fix Fail @ {pass_k}: = {100 * data['fail_fail'] / (data['pass_pass'] + data['pass_fail'] + data['fail_pass'] + data['fail_fail']):.2f}")
    if fix_only_mode:
        validity_avg = 0
        non_zero_ratio = 0
    tex_str += f' & {100 * fixed_rate:.2f}\% & {100 * validity_avg:.2f}\% & {100 * non_zero_ratio:.2f}\% '
print(tex_str + " \\\\")

print(f'pass_pass = {100 *pass_pass/sum_number:.2f}, pass_fail = {100 *pass_fail/sum_number:.2f}, fail_pass = {100 *fail_pass/sum_number:.2f}, fail_fail = {100 *fail_fail/sum_number:.2f}')
# pass_pass = 4.86, pass_fail = 1.77, fail_pass = 21.35, fail_fail = 72.02
# pass_pass = 38.95, pass_fail = 13.62, fail_pass = 32.40, fail_fail = 15.02
# pass_pass = 10.97, pass_fail = 2.80, fail_pass = 26.58, fail_fail = 59.65
# pass_pass = 47.28, pass_fail = 11.34, fail_pass = 29.38, fail_fail = 12.00
# pass_pass = 8.03, pass_fail = 2.14, fail_pass = 58.69, fail_fail = 31.15
# pass_pass = 55.67, pass_fail = 11.19, fail_pass = 22.53, fail_fail = 10.60