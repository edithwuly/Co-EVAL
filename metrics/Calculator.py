def calculate(expression, response):
    try:
        result = eval(expression)
        if str(result) == response:
            return 1
        else:
            return 0
    except Exception as e:
        return f"Error: {str(e)}"