#!/usr/bin/env python3
"""
rename_csv_seq.py

将目录下形如 <student>_<num1>_<num2>.csv 的文件，
按 <student> 分组，并按 (num1, num2) 排序后
重命名为 <student>_<seq>.csv（每个 student 从 1 开始）。

用法：
    python rename_csv_seq.py /path/to/folder --dry-run
"""

import os
import re
import argparse

PATTERN = re.compile(r'^(?P<student>.+)_(?P<n1>\d+)_(?P<n2>\d+)\.csv$', re.IGNORECASE)

def collect_files(folder):
    mapping = {}
    for name in os.listdir(folder):
        m = PATTERN.match(name)
        if not m:
            continue
        student = m.group('student')
        n1 = int(m.group('n1'))
        n2 = int(m.group('n2'))
        mapping.setdefault(student, []).append((name, n1, n2))
    return mapping

def build_rename_plan(folder):
    mapping = collect_files(folder)
    plan = []  # list of tuples (orig_path, final_path)
    for student, files in mapping.items():
        # sort by the numeric parts (n1, n2)
        files_sorted = sorted(files, key=lambda x: (x[1], x[2]))
        for idx, (orig_name, n1, n2) in enumerate(files_sorted, start=1):
            seq = f"{idx}"
            new_name = f"{student}_{seq}.csv"
            plan.append((os.path.join(folder, orig_name), os.path.join(folder, new_name)))
    return plan

def execute_plan(plan, dry_run=True):
    if not plan:
        print("未找到匹配的文件，退出。")
        return
    print(f"准备处理 {len(plan)} 个文件。dry_run={dry_run}")
    for orig, new in plan:
        print(f"{os.path.basename(orig)} -> {os.path.basename(new)}")
    if dry_run:
        print("dry-run 模式，未实际重命名。若确认无误，请去掉 --dry-run 选项重新运行。")
        return

    # 两步重命名以避免覆盖冲突：先重命名为临时文件，再改为最终文件名
    tmp_names = []
    try:
        for i, (orig, new) in enumerate(plan):
            tmp = f"{orig}.rename_tmp_{i}"
            os.rename(orig, tmp)
            tmp_names.append((tmp, new))
        # 再从临时名改回目标名
        for tmp, new in tmp_names:
            if os.path.exists(new):
                raise FileExistsError(f"目标文件已存在：{new}")
            os.rename(tmp, new)
    except Exception as e:
        print("重命名过程中发生错误：", e)
        # 尝试回滚未完成的操作（把临时的改回原名）
        for tmp, new in tmp_names:
            if os.path.exists(tmp):
                orig = tmp.rsplit(".rename_tmp_", 1)[0]
                try:
                    os.rename(tmp, orig)
                except Exception:
                    pass
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="要处理的文件夹路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印重命名计划，不做实际重命名")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print("错误：指定路径不是目录。")
        return

    plan = build_rename_plan(folder)
    if not plan:
        print("未发现符合模式的文件待重命名。")
        return

    execute_plan(plan, dry_run=args.dry_run)
    if not args.dry_run:
        print("重命名完成。")

if __name__ == "__main__":
    main()
