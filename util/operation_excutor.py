import os

from database.schema import Operation

def excute_operation(op: Operation) -> None:
    type_name = op.operation_type.type_name
    extra_data = op.extra_data
    def exec_file() -> None:
        os.system('start ""  "{}"'.format(extra_data))
    def short_cut() -> None:
        pass
    def run_cmd() -> None:
        os.system(extra_data)
    func_mapping = {
        "执行命令": run_cmd,
        "快捷键": short_cut,
        "运行程序": exec_file,
    }
    if type_name not in func_mapping:
        return
    func = func_mapping[type_name]
    func()