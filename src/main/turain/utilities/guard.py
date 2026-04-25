from utilities import helper_method


@helper_method
def check_arguments(**kwargs):
    for _type, (value, predicate) in kwargs:
        if value is None:
            print(f"A value hasn't been provided for the parameter : {_type, value, predicate}, Skipping...")
            return
        if type(value) is not _type:
            raise TypeError(f"{value} must be {_type}")
        if predicate is not None and not predicate["predicate"]:
            raise ValueError(predicate["error_message"])

@helper_method
def check_positive_integer(*args):
    for var in args:
        check_arguments(
            int,
            (
                var,
                {
                    "predicate": lambda output_width: output_width < 1,
                    "error_message": f"{var.__name__} must be a positive integer",
                },
            )
        )
