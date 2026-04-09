import datasets

# 加载您本地的 'default' 数据集
local_dataset = datasets.load_dataset('./datasets/wikitext', 'default', split='test')
print("--- 本地 'default' 数据集信息 ---")
print(local_dataset)
print(f"特征: {local_dataset.features}")
print(f"行数: {local_dataset.num_rows}")

print("\n" + "="*50 + "\n")

# 直接从网络加载原始的 'wikitext-2-raw-v1' 数据集进行对比
online_dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
print("--- Hugging Face 网络 'wikitext-2-raw-v1' 数据集信息 ---")
print(online_dataset)
print(f"特征: {online_dataset.features}")
print(f"行数: {online_dataset.num_rows}")
