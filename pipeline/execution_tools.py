from langchain.tools import tool
import numpy as np

@tool
def add(arg1: float, arg2 : float) -> str:
    """This tool can be used to do simple addition. This must accept two numerical arguments."""
    try:
       answer = arg1 + arg2
       return answer
    except Exception as e:
        return f"Error: {e}"

@tool
def subtract(arg1: float, arg2 : float):
    """This tool can be used to do simple subtraction. This must acceptt two numerical arguments."""
    try:
       answer = arg1 - arg2
       return answer
    except Exception as e:
        return f"Error: {e}"


@tool
def multiply(arg1: float, arg2 : float):
    """This tool can be used to do simple multiplication. This must accept two numerical arguments."""
    try:
       answer = arg1 * arg2
       return answer
    except Exception as e:
        return f"Error: {e}"

@tool
def divide(arg1: float, arg2 : float):
    """This tool can be used to do simple division. This must accept two numerical arguments."""
    try:
       answer = arg1 / arg2
       return answer
    except Exception as e:
        return f"Error: {e}"

@tool
def convert_to_percentage(arg1: float) -> str:
    """This tool can be used to convert a numerical value to the string representation as percentage."""
    try:
       pct = np.round(arg1*100, 4)
       return f"{pct}%"
    except Exception as e:
        return f"Error: {e}"

