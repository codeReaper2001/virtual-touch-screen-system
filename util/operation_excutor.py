from typing import Optional
import os
import webbrowser
import pyautogui

from database.schema import Operation

def excute_operation(op: Optional[Operation]) -> None:
    if not op:
        return
    type_name = op.operation_type.type_name
    extra_data = op.extra_data
    def exec_file() -> None:
        os.system('start ""  "{}"'.format(extra_data))
    def short_cut() -> None:
        keys_str = extra_data.lower()
        keys = keys_str.split("+")
        print(keys)
        pyautogui.hotkey(*keys)
    def run_cmd() -> None:
        os.system(extra_data)
    def browse_url() -> None:
        webbrowser.open_new_tab(extra_data)

    func_mapping = {
        "执行命令": run_cmd,
        "快捷键": short_cut,
        "运行程序": exec_file,
        "打开网页": browse_url,
    }
    if type_name not in func_mapping:
        return
    func = func_mapping[type_name]
    func()