def calculate_cyclomatic_complexity(source_code):
    complexity = 0
    complexity += source_code.count("if")
    complexity += source_code.count("for")
    complexity += source_code.count("while")
    complexity += source_code.count("and")
    complexity += source_code.count("or")
    complexity += source_code.count("except")
    return complexity