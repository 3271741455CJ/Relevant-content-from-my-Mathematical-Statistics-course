import numpy as np
import pandas as pd
import os


def calculate_epps_pulley_statistic(data):
    """计算爱泼斯-普利检验（EP检验）统计量 T_EP"""
    x = np.array(data, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < 8:
        raise ValueError(f"EP检验要求样本量 n >= 8，当前有效样本量为 {n}")

    # 计算样本均值与二阶中心矩 m2
    x_bar = np.mean(x)
    m2 = np.mean((x - x_bar) ** 2)

    if m2 == 0:
        raise ValueError("样本方差为0，数据全部相同，无法进行EP检验")

    A = np.sum(np.exp(-(x - x_bar) ** 2 / (4 * m2)))

    diff_matrix = x[:, np.newaxis] - x
    upper_triangle_indices = np.triu_indices(n, k=1)
    B = np.sum(np.exp(-(diff_matrix[upper_triangle_indices] ** 2) / (2 * m2)))

    T_EP = 1 + n / np.sqrt(3) + (2 / n) * B - np.sqrt(2) * A

    print(f"\n========================================")
    print(f"最终用于计算的有效样本量 (n) : {n}")
    print(f"检验统计量 (T_EP): {T_EP:.6f}")
    print(f"========================================\n")

    return T_EP


def load_data_from_user():
    """读取用户指定路径的文件数据，并提供预览和选择"""
    file_path = input("请输入数据文件的完整路径:\n> ").strip("\"'")

    if not os.path.exists(file_path):
        print("错误：找不到该文件，请检查路径是否正确！")
        return None

    try:
        ext = os.path.splitext(file_path)[-1].lower()

        # 使用 sep=None 和 engine='python' 开启自动嗅探，完美识别 \t、逗号或空格
        if ext in ['.csv', '.txt']:
            df = pd.read_csv(file_path, sep=None, engine='python', header=None)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, header=None)
        else:
            print("错误：不支持的文件格式。")
            return None

        print(f"\n-> 文件读取成功！检测到原始表格形状为: {df.shape[0]} 行 x {df.shape[1]} 列")

        # 如果表格有多列，询问用户如何处理
        if df.shape[1] > 1:
            print("注意：你的文件中包含多列数据。")
            choice = input(
                "请选择提取方式:\n [1] 仅提取第 1 列 (推荐，可排除右侧的其它变量)\n [2] 提取整个表格的所有数据 (仅当你的数据呈矩阵状分布时使用)\n> ").strip()

            if choice == '1':
                raw_data = df.iloc[:, 0].values
            else:
                raw_data = df.values.flatten()
        else:
            raw_data = df.values.flatten()

        # 强制转换为数值类型，无法转换的文本（如表头）会变成 NaN 并被剔除
        data = pd.to_numeric(raw_data, errors='coerce')
        data = data[~np.isnan(data)]

        print(f"-> 数据清洗完毕，提取到 {len(data)} 个有效数值。")
        if len(data) > 0:
            print(f"-> 数据预览 (前5个): {data[:5]}")

        return data

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


if __name__ == "__main__":
    user_data = load_data_from_user()
    if user_data is not None:
        calculate_epps_pulley_statistic(user_data)