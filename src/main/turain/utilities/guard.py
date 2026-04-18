from utilities import helper_method


@helper_method
def check_arguments(**kwargs):
    for _type, (value, predicate) in kwargs:
        if type(value) is not _type:
            raise TypeError(f"{value} must be {_type}")
        if predicate is not None and not predicate["predicate"]:
            raise ValueError(predicate["error_message"])
