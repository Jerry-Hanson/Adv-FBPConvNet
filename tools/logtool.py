import inspect

# ANSI颜色代码
YELLOW = '\033[93m'  # 黄色
GREEN = '\033[92m'  # 绿色
BLUE = '\033[94m'  # 蓝色
RED = '\033[91m'  # 红色
ENDC = '\033[0m'   # 恢复默认颜色

def printParam(*args, **kwargs):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    print(f"{YELLOW}File: {filename}, Line: {lineno}{ENDC}")
    
    for arg in args:
        var_name = [var for var, val in frame.f_locals.items() if val is arg]
        if var_name:
            var_name = var_name[0]
            value = arg
            print(f"{RED}Variable Name: {var_name}, Value: {value}{ENDC}")
            
    for key, value in kwargs.items():
        var_name = key
        value = value
        print(f"{RED}Variable Name: {var_name}, Value: {value}{ENDC}")
