def core_method(func):
    func.__core_method__ = True
    return func