import matplotlib.pyplot as plt


def format_reward():
    # 示例数据
    x = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
    y1 = [0, 0.8, 0.9, 0.93, 0.95, 0.954, 0.967, 0.97]
    y2 = [0, 0.82, 0.95, 0.955, 0.962, 0.96, 0.968, 0.96]

    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({
        'legend.fontsize': 20,
    })

    # 绘制两条折线
    plt.plot(x, y1, label='Train', marker='o', markersize=10)
    plt.plot(x, y2, label='Test', marker='s', markersize=10, markerfacecolor='green', markeredgecolor='green',
             color='green')
    plt.title('(a). Format Reward', fontsize=20)

    # 添加标题和标签
    plt.xlabel('Steps', fontsize=20)
    # plt.ylabel('Code Repair Reward', fontsize=16)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True, alpha=0.5)
    # 设置边距：最小化边距，但仍留出空间显示坐标轴标签
    fig.subplots_adjust(left=0, right=1.0, top=0.95, bottom=0.08)
    plt.savefig("./data/pdf/format_reward.pdf", format="pdf", bbox_inches='tight')

    # 显示图像
    # plt.show()


def code_repair_reward():
    # 示例数据
    x = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
    y1 = [0, 0.39, 0.5, 0.56, 0.6, 0.67, 0.7, 0.697]
    y2 = [0, 0.41, 0.57, 0.585, 0.5997, 0.63, 0.65, 0.66]

    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({
        'legend.fontsize': 20,
    })

    # 绘制两条折线
    plt.plot(x, y1, label='Train', marker='o', markersize=10)
    plt.plot(x, y2, label='Test', marker='s', markersize=10, markerfacecolor='green', markeredgecolor='green',
             color='green')
    plt.title('(b). Code Repair Reward', fontsize=20)

    # 添加标题和标签
    plt.xlabel('Steps', fontsize=20)
    # plt.ylabel('Code Repair Reward', fontsize=16)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True, alpha=0.5)
    # 设置边距：最小化边距，但仍留出空间显示坐标轴标签
    fig.subplots_adjust(left=0, right=1.0, top=0.95, bottom=0.08)
    plt.savefig("./data/pdf/code_repair_reward.pdf", format="pdf", bbox_inches='tight')

    # 显示图像
    # plt.show()


def generation_test_reward():
    # 示例数据
    x = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
    y1 = [0, 0.26, 0.32, 0.35, 0.3513, 0.42, 0.39, 0.43]
    y2 = [0, 0.24, 0.33, 0.34, 0.356, 0.4, 0.38, 0.42]

    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({
        'legend.fontsize': 20,
    })

    # 绘制两条折线
    plt.plot(x, y1, label='Train', marker='o', markersize=10)
    plt.plot(x, y2, label='Test', marker='s', markersize=10, markerfacecolor='green', markeredgecolor='green',
             color='green')
    plt.title('(c). Test Generation Reward', fontsize=20)

    # 添加标题和标签
    plt.xlabel('Steps', fontsize=20)
    # plt.ylabel('Code Repair Reward', fontsize=16)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True, alpha=0.5)
    # 设置边距：最小化边距，但仍留出空间显示坐标轴标签
    fig.subplots_adjust(left=0, right=1.0, top=0.95, bottom=0.08)
    plt.savefig("./data/pdf/test_generation_reward.pdf", format="pdf", bbox_inches='tight')

    # 显示图像
    # plt.show()


def response_length():
    # 示例数据
    x = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
    y1 = [350, 290, 250, 257, 267, 251, 256, 244]
    y2 = [340, 280, 211, 215, 222, 225, 229, 230]

    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({
        'legend.fontsize': 20,
    })

    # 绘制两条折线
    plt.plot(x, y1, label='Train', marker='o', markersize=10)
    plt.plot(x, y2, label='Test', marker='s', markersize=10, markerfacecolor='green', markeredgecolor='green',
             color='green')
    plt.title('(d). Response Length', fontsize=20)

    # 添加标题和标签
    plt.xlabel('Steps', fontsize=20)
    # plt.ylabel('Code Repair Reward', fontsize=16)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True, alpha=0.5)
    # 设置边距：最小化边距，但仍留出空间显示坐标轴标签
    fig.subplots_adjust(left=0, right=1.0, top=0.95, bottom=0.08)
    plt.savefig("./data/pdf/response_length.pdf", format="pdf", bbox_inches='tight')

    # 显示图像
    # plt.show()


# pass_pass = 4.86, pass_fail = 1.77, fail_pass = 21.35, fail_fail = 72.02
# pass_pass = 38.95, pass_fail = 13.62, fail_pass = 32.40, fail_fail = 15.02
# pass_pass = 10.97, pass_fail = 2.80, fail_pass = 26.58, fail_fail = 59.65
# pass_pass = 47.28, pass_fail = 11.34, fail_pass = 29.38, fail_fail = 12.00
# pass_pass = 8.03, pass_fail = 2.14, fail_pass = 58.69, fail_fail = 31.15
# pass_pass = 55.67, pass_fail = 11.19, fail_pass = 22.53, fail_fail = 10.60
def pie_chart():
    # 示例数据（6 个模型）
    model_names = [
        "Qwen2.5-Coder-1.5B(Vanilla)",
        "Qwen2.5-Coder-1.5B(RL)",
        "Qwen2.5-Coder-3B(Vanilla)",
        "Qwen2.5-Coder-3B(RL)",
        "Qwen-4B(Vanilla)",
        "Qwen-4B(RL)"
    ]

    labels = ['fix w/ test', 'fix w/o test', 'fail w/ test', 'fail w/o test']
    sizes_list = [
        [4.86, 21.35, 1.77, 72.02],
        [38.95, 32.4, 13.62, 15.02],
        [10.97, 26.58, 2.8, 59.65],
        [47.28, 29.38, 11.34, 12],
        [8.03, 58.69, 2.14, 31.15],
        [55.67, 22.53, 11.19, 10.6]
    ]  # 每个模型对应的四个部分占比

    # 预定义多个配色方案
    color_palettes = [
        ['#74c476', '#a6bddb', '#fde0ef', '#fcd0d1'],
        ['#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'],
        ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'],
        ['#f7fbff', '#08306b', '#0868ac', '#4393c3']
    ]

    # 创建 1 行 6 列 的子图布局
    fig, axes = plt.subplots(1, 6, figsize=(24, 4), constrained_layout=True)

    for i, (ax, name, sizes) in enumerate(zip(axes, model_names, sizes_list)):
        colors = ['#74c476', '#a6bddb', '#fde0ef', '#fcd0d1']

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct='%.1f%%',
            startangle=90,
            colors=colors,
            textprops={'size': 'smaller'}
        )
        ax.set_title(name, fontsize=10, pad=5)
        ax.axis('equal')

        # 添加图例到每个子图的右上角外部
        ax.legend(wedges, labels,
                  loc="upper right",
                  bbox_to_anchor=(1.2, 1),
                  prop={"size": 8},
                  title_fontsize='8')

    # 保存为 PDF 文件
    fig.savefig("./data/pdf/six_pie_charts.pdf", format='pdf', bbox_inches='tight')

    # 显示图形
    plt.show()


# 1.5b [0.8125,0.8571,0.8839,0.8929,0.8929,0.9018,0.9018,0.9018] [0.6615,0.7432,0.7821,0.8016,0.8132,0.8249,0.8366,0.8521] [0.4667,0.5607,0.6085,0.6325,0.6530,0.6667,0.6735,0.6786] [0.3985,0.5025,0.5569,0.5767,0.6015,0.6213,0.6312,0.6460]
# 3b [0.8571,0.9018,0.9196,0.9375,0.9375,0.9375,0.9464,0.9554] [0.69650.7860,0.8054,0.8288,0.8444,0.8482,0.8521,0.8716] [0.5385,0.5846,0.6205,0.6410,0.6564,0.6718,0.6786,0.6872] [0.5,0.6163,0.6609,0.6931,0.7054,0.7203,0.7327,0.7401]
# 4b [0.8661,0.9018,0.9196,0.9286,0.9464,0.9643,0.9732,0.9732] [0.7198,0.7821,0.8093,0.8288,0.8366,0.8405,0.8405,0.8444] [0.5419,0.6171,0.6393,0.6701,0.6786,0.6923,0.7026,0.7145] [0.5173,0.6015,0.6510,0.6782,0.7030,0.7129,0.7302,0.7450]


def scaling_humaneval():
    # 示例数据
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y1 = [0.8125, 0.8571, 0.8839, 0.8929, 0.8929, 0.9018, 0.9018, 0.9018]
    y2 = [0.8571, 0.9018, 0.9196, 0.9375, 0.9375, 0.9375, 0.9464, 0.9554]
    y3 = [0.8661, 0.9018, 0.9196, 0.9286, 0.9464, 0.9643, 0.9732, 0.9732]

    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({
        'legend.fontsize': 20,
    })

    plt.plot(x, y1, label='Repair-R1(Qwen2.5-Coder-1.5B-Instruct)', marker='o', markersize=10,color='#5D3A9B')
    plt.plot(x, y2, label='Repair-R1(Qwen2.5-Coder-3B-Instruct)', marker='s', markersize=10,color='#9E76C4')
    plt.plot(x, y3, label='Repair-R1(Qwen3-4B)', marker='^', markersize=10, color='#CAB2D6')
    # plt.title('(a). HumanEval', fontsize=20)

    # 添加标题和标签
    plt.xlabel('Bugfix@K', fontsize=20)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True, alpha=0.5)
    # 设置边距：最小化边距，但仍留出空间显示坐标轴标签
    fig.subplots_adjust(left=0, right=1.0, top=0.95, bottom=0.08)
    plt.savefig("./data/pdf/scale_humaneval.pdf", format="pdf", bbox_inches='tight')

    # 显示图像
    plt.show()

def scaling_mbpp():
    # 示例数据
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y1 = [0.6615,0.7432,0.7821,0.8016,0.8132,0.8249,0.8366,0.8521]
    y2 = [0.6965,0.7860,0.8054,0.8288,0.8444,0.8482,0.8521,0.8716]
    y3 = [0.7198,0.7821,0.8093,0.8288,0.8366,0.8405,0.8405,0.8444]
    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({
        'legend.fontsize': 20,
    })

    plt.plot(x, y1, label='Repair-R1(Qwen2.5-Coder-1.5B-Instruct)', marker='o', markersize=10,color='#5D3A9B')
    plt.plot(x, y2, label='Repair-R1(Qwen2.5-Coder-3B-Instruct)', marker='s', markersize=10,color='#9E76C4')
    plt.plot(x, y3, label='Repair-R1(Qwen3-4B)', marker='^', markersize=10, color='#CAB2D6')
    # plt.title('(b). MBPP', fontsize=20)

    # 添加标题和标签
    plt.xlabel('Bugfix@K', fontsize=20)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True, alpha=0.5)
    # 设置边距：最小化边距，但仍留出空间显示坐标轴标签
    fig.subplots_adjust(left=0, right=1.0, top=0.95, bottom=0.08)
    plt.savefig("./data/pdf/scale_mbpp.pdf", format="pdf", bbox_inches='tight')

    # 显示图像
    plt.show()

def scaling_codeforces():
    # 示例数据
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y1 = [0.4667,0.5607,0.6085,0.6325,0.6530,0.6667,0.6735,0.6786]
    y2 = [0.5385,0.5846,0.6205,0.6410,0.6564,0.6718,0.6786,0.6872]
    y3 = [0.5419,0.6171,0.6393,0.6701,0.6786,0.6923,0.7026,0.7145]
    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({
        'legend.fontsize': 20,
    })

    plt.plot(x, y1, label='Repair-R1(Qwen2.5-Coder-1.5B-Instruct)', marker='o', markersize=10,color='#5D3A9B')
    plt.plot(x, y2, label='Repair-R1(Qwen2.5-Coder-3B-Instruct)', marker='s', markersize=10,color='#9E76C4')
    plt.plot(x, y3, label='Repair-R1(Qwen3-4B)', marker='^', markersize=10, color='#CAB2D6')
    # plt.title('(c). CodeForces', fontsize=20)

    # 添加标题和标签
    plt.xlabel('Bugfix@K', fontsize=20)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True, alpha=0.5)
    # 设置边距：最小化边距，但仍留出空间显示坐标轴标签
    fig.subplots_adjust(left=0, right=1.0, top=0.95, bottom=0.08)
    plt.savefig("./data/pdf/scale_codeforces.pdf", format="pdf", bbox_inches='tight')

    # 显示图像
    plt.show()


def scaling_codeContests():
    # 示例数据
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y1 =[0.3985,0.5025,0.5569,0.5767,0.6015,0.6213,0.6312,0.6460]
    y2 = [0.5,0.6163,0.6609,0.6931,0.7054,0.7203,0.7327,0.7401]
    y3 = [0.5173,0.6015,0.6510,0.6782,0.7030,0.7129,0.7302,0.7450]
    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.rcParams.update({
        'legend.fontsize': 20,
    })

    plt.plot(x, y1, label='Repair-R1(Qwen2.5-Coder-1.5B-Instruct)', marker='o', markersize=10, color='#5D3A9B')
    plt.plot(x, y2, label='Repair-R1(Qwen2.5-Coder-3B-Instruct)', marker='s', markersize=10, color='#9E76C4')
    plt.plot(x, y3, label='Repair-R1(Qwen3-4B)', marker='^', markersize=10, color='#CAB2D6')
    # plt.title('(d). CodeForces', fontsize=20)

    # 添加标题和标签
    plt.xlabel('Bugfix@K', fontsize=20)
    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True, alpha=0.5)
    # 设置边距：最小化边距，但仍留出空间显示坐标轴标签
    fig.subplots_adjust(left=0, right=1.0, top=0.95, bottom=0.08)
    plt.savefig("./data/pdf/scale_codecontests.pdf", format="pdf", bbox_inches='tight')

    # 显示图像
    plt.show()


if __name__ == '__main__':
    # format_reward()
    # code_repair_reward()
    # generation_test_reward()
    # response_length()
    # pie_chart()
    scaling_humaneval()
    scaling_mbpp()
    scaling_codeforces()
    scaling_codeContests()
