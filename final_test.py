import os
import shutil
from datasets import load_dataset

# 决定性的步骤：我们把目标路径设置在您的用户主目录(~)
# 这几乎可以保证我们是在一个标准的本地文件系统上操作
# 我们将创建一个名为 'hf_data_test' 的文件夹来存放一切
# 完全绕开项目路径和 /gammadisk
home_dir = os.path.expanduser("~")
test_base_path = os.path.join(home_dir, 'hf_data_test')
save_path = os.path.join(test_base_path, 'wikitext')
cache_path = os.path.join(test_base_path, 'cache')

print(f"--- 决定性测试：将在您的用户主目录进行读写 ---")
print(f"目标数据目录: {save_path}")

try:
    # 1. 彻底清场：确保每次运行都是全新的开始，删除旧的测试文件夹
    if os.path.exists(test_base_path):
        print(f"发现旧的测试目录，正在删除: {test_base_path}")
        shutil.rmtree(test_base_path)
    print("旧目录清理完毕。")

    # 2. 创建一个全新的、干净的缓存和保存目录
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(cache_path, exist_ok=True)
    print("已创建全新的、干净的测试目录。")
    
    # 3. 从网络加载，并将下载缓存也指定到我们的“安全区”
    print("\n正在从 Hugging Face 下载数据集...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=cache_path)
    print("下载成功！内存中数据集信息：")
    print(dataset)
    
    # 4. 保存到磁盘我们的“安全区”
    print(f"\n正在将数据集保存到: {save_path}")
    dataset.save_to_disk(save_path)
    print("`save_to_disk` 命令执行完毕。")
    
    # 5. 决定性的一步：立刻从磁盘加载回来进行验证
    print(f"\n正在从磁盘重新加载数据集进行验证...")
    reloaded_dataset = load_dataset(save_path)
    print("重新加载成功！磁盘上的数据集信息：")
    print(reloaded_dataset)
    
    # 6. 最终断言：检查行数是否正确
    assert reloaded_dataset['test'].num_rows == 4358
    assert reloaded_dataset['train'].num_rows == 36718
    
    print("\n\n" + "="*80)
    print("🎉🎉🎉 恭喜！测试完全成功！🎉🎉🎉")
    print("这证明：")
    print("1. 数据集可以被正确地下载、保存和加载。")
    print(f"2. 问题根源在于您原来的项目路径 (位于 /gammadisk)，它所在的文件系统与 `datasets` 库不兼容。")
    print("\n下一步解决方案:")
    print(f"   A. 将您的整个项目都移动到 gammadisk 之外的位置，例如 '~/FlatQuant'。")
    print(f"   B. 或者，让您的系统管理员检查 /gammadisk 的挂载配置。")
    print("="*80 + "\n")

except Exception as e:
    print(f"\n\n" + "x"*80)
    print("... 测试再次失败 ...")
    print(f"错误信息: {e}")
    print("\n如果这个测试依然失败，说明这台服务器的环境非常特殊。")
    print("请联系您的系统管理员，并向他们说明：")
    print("'Huggingface datasets library (using pyarrow) fails to save and reload datasets even in the user's home directory.'")
    print("x"*80 + "\n")
