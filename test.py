import os

# 配置：要禁止展开的目录列表（不区分大小写）
EXCLUDE_DIRS = {"dataset"}

def print_directory_structure(path, prefix="", is_last=True):
    try:
        items = os.listdir(path)
        items.sort()

        for i, item in enumerate(items):
            item_path = os.path.join(path, item)
            is_last_item = (i == len(items) - 1)

            current_prefix = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            print(prefix + current_prefix + item)

            # 核心：只要是目录且在排除列表中（不区分大小写），绝对不递归
            if os.path.isdir(item_path):
                if item.lower() not in EXCLUDE_DIRS:
                    print_directory_structure(item_path, next_prefix, is_last_item)

    except PermissionError:
        print(prefix + "└── [权限不足]")
    except Exception as e:
        print(prefix + f"└── [错误: {str(e)}]")


def main():
    print("目录结构生成器")
    print("=" * 50)

    while True:
        path = input("\n输入路径（q退出）: ").strip()
        if path.lower() == 'q':
            print("退出")
            break

        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            print(f"路径不存在: {abs_path}")
            continue

        print(f"\n{os.path.basename(abs_path)}/")
        print_directory_structure(abs_path)
        print("=" * 50)


if __name__ == "__main__":
    main()