import inspect

from cutex.src_module import previous_frame_arg_lineno

def my_function(a, b, c):
    lineno = previous_frame_arg_lineno(b)
    print(f"Line number of 'b' argument value in the previous frame: {lineno}")

x = 2
y = 1

my_function(1, 
            x, 3)
