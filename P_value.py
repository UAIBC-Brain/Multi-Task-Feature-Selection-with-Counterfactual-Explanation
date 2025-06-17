import pandas as pd
from scipy.stats import ttest_ind

# 读取数据
df = pd.read_csv('G:\\AWSLwork\\Awsl9\\Result-noindex\\Xiangya_selected.csv')

# 初始化一个列表存储 P 值及其索引
p_values = []

# 遍历每一列（除了最后一列，因为最后一列是标签）
for idx, column in enumerate(df.columns[:-1], start=1):  # start=1 让索引从 1 开始
    # 根据标签分组数据
    group_0 = df[df.iloc[:, -1] == 0][column]
    group_1 = df[df.iloc[:, -1] == 1][column]

    # 计算 t 检验的 P 值
    t_stat, p = ttest_ind(group_0, group_1, equal_var=False)

    # 存储 (列索引, 列名, p 值) 的元组
    p_values.append((idx, column, p))

# 输出每一列的 P 值
for idx, column, p_value in p_values:
    print(f'Column {idx} ({column}) p-value: {p_value}')

# 找到最小的 5 个 P 值及其对应的列索引和列名
top_5_p_values = sorted(p_values, key=lambda x: x[2])[:5]  # 按 p 值排序

print("\nTop 5 smallest p-values with column indices:")
for rank, (idx, column, p_value) in enumerate(top_5_p_values, start=1):
    print(f'Top {rank}: Column {idx} ({column}) - p-value: {p_value}')
