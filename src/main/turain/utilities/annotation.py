def core_method(func):
    func.__core_method__ = True
    return func

def helper_method(func):
    func.__helper_method__ = True
    return func