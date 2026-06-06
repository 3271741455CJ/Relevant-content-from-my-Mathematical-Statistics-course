import os
import numpy as np
import scipy.stats as stats
import pandas as pd


def manual_ks_test(data, cdf_func):
    """
    根据柯尔莫哥洛夫检验原理计算 K-S 统计量 Dn
    """
    n = len(data)

    # 1. 找到有序样本点
    x_sorted = np.sort(data)

    # 2. 计算理论分布函数在各个样本点的值
    f0_x = cdf_func(x_sorted)

    # 3. 构造经验分布函数 Fn(x) 的阶跃边界
    i_array = np.arange(1, n + 1)
    ecdf_lower = (i_array - 1) / n
    ecdf_upper = i_array / n

    # 4. 计算绝对差值并求最大距离 Dn
    diff_lower = np.abs(f0_x - ecdf_lower)
    diff_upper = np.abs(f0_x - ecdf_upper)
    dn = np.max([np.max(diff_lower), np.max(diff_upper)])

    return dn


def load_table_data(file_path):
    """
    读取分布在多行多列的表格数据，并展平为一维数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}，请检查路径。")

    ext = os.path.splitext(file_path)[-1].lower()

    try:
        # header=None 表示数据没有列名，直接从第一行开始读
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, header=None)
        elif ext == '.csv':
            df = pd.read_csv(file_path, header=None)
        elif ext in ['.txt', '.dat']:
            # 对于以空格或制表符分隔的文本文件
            df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        else:
            raise ValueError(f"暂不支持的文件格式: {ext}，请提供 .xlsx, .csv 或 .txt 文件。")

        # 核心修改点：
        # 1. .values 获取二维数组
        # 2. .flatten() 将多行多列展平为一维数组
        data_1d = df.values.flatten()

        # 3. 剔除空白单元格产生的 NaN (Not a Number) 值
        # pd.isna() 会找出所有的缺失值，~ 表示取反（即保留非缺失值）
        clean_data = data_1d[~pd.isna(data_1d)]

        return clean_data

    except Exception as e:
        raise Exception(f"读取文件时发生错误: {str(e)}")


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("欢迎使用 K-S 检验工具 (兼容二维表格数据)")
    print("=" * 50)

    # 如果你还没有安装 openpyxl 库来读取 excel，程序可能会提示你安装：
    # 可以通过 pip install openpyxl pandas scipy 解决
    user_input = input("请输入表格文件的完整路径 (支持 .xlsx, .csv, .txt): ").strip()
    user_input = user_input.strip('\"').strip('\'')

    try:
        # 加载数据
        print(f"\n正在加载数据: {user_input} ...")
        sample_data = load_table_data(user_input)

        # 打印一下提取出的数据预览，确保读取正确
        print(f"成功读取数据！有效样本量 n = {len(sample_data)}")
        print(f"数据前5个值预览: {sample_data[:5]}\n")

        # 动态计算均值和标准差，用于构建正态分布参数
        mean_val = np.mean(sample_data)
        std_val = np.std(sample_data, ddof=1)

        custom_norm_cdf = lambda x: stats.norm.cdf(x, loc=mean_val, scale=std_val)

        # 执行检验
        dn_manual = manual_ks_test(sample_data, custom_norm_cdf)
        ks_result_scipy = stats.kstest(sample_data, 'norm', args=(mean_val, std_val))

        # 输出结果
        print("-" * 40)
        print("K-S 检验计算结果")
        print("-" * 40)
        print(f"拟合分布: 正态分布 N({mean_val:.4f}, {std_val:.4f}^2)")
        print(f"手动公式计算 Dn        : {dn_manual:.6f}")
        print(f"Scipy库计算 Dn         : {ks_result_scipy.statistic:.6f}")
        print(f"Scipy 提供的 p-value   : {ks_result_scipy.pvalue:.6f}")

        # 假设检验判别
        alpha = 0.05
        if ks_result_scipy.pvalue > alpha:
            print(f"\n结论: p-value > {alpha}，不能拒绝原假设，该样本数据符合正态分布。")
        else:
            print(f"\n结论: p-value <= {alpha}，拒绝原假设，该样本数据不符合正态分布。")

    except Exception as e:
        print(f"\n程序运行出错: {e}")