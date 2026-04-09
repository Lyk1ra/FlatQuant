from datasets import load_dataset
import os

# 确保脚本下载的是 FlatQuant 代码需要的 'wikitext-2-raw-v1' 配置
dataset_name = 'wikitext'
config_name = 'wikitext-2-raw-v1'

# 数据集的本地保存路径
save_path = './my_data_storage/wikitext'

print(f"开始从 Hugging Face 下载 {dataset_name} ({config_name})...")

try:
    # 1. 从网络加载完整的数据集（包含train, test, validation所有部分）
    full_dataset = load_dataset(dataset_name, config_name)
    print("\n下载完成。数据集信息如下：")
    print(full_dataset)

    # 2. 将完整数据集保存到本地磁盘
    print(f"\n正在将数据集保存到本地路径: {save_path}")
    full_dataset.save_to_disk(save_path)
    
    print(f"\n*** 数据集已成功保存到 {save_path}! ***")
    
    # 3. (自动验证) 重新加载本地数据，确保其完好无损
    print("\n正在验证已保存的本地数据...")
    reloaded_dataset = load_dataset(save_path, 'default') # 加载本地数据时，配置名就是 'default'
    print("本地数据重载成功！验证信息如下：")
    print(reloaded_dataset)

    # 检查测试集的行数是否正确
    assert reloaded_dataset['test'].num_rows == 4358
    print("\n行数验证成功！您的本地数据集现在是完整且正确的！")

except Exception as e:
    print(f"\n处理过程中发生错误: {e}")
    print("请检查您的网络连接或磁盘空间，然后重试脚本。")
