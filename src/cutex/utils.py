def ceildiv(a:int, b:int) -> int:
    """a faster and more accurate version of `math.ceil(a / b)`

    Reference:
        https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    """
    return -(int(a) // -int(b))
