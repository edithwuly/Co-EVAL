import re

def calculate_cognitive_complexity(code):
    lines = code.strip().split('\n')
    complexity = 0
    nesting_level = 0

    control_flow_keywords = ['if', 'else', 'for', 'while', 'do', 'switch', 'case', 'catch', 'try']
    break_continue_keywords = ['break', 'continue']

    for line in lines:
        # Strip comments and leading/trailing whitespace
        line = line.split('//')[0].strip()
        if not line:
            continue

        # Check for control flow statements
        for keyword in control_flow_keywords:
            if re.search(rf'\b{keyword}\b', line) and not line.endswith(';'):
                # Base increment for control flow
                complexity += 1

                # Additional increment for nesting
                complexity += nesting_level

                # Increase nesting level for blocks that continue
                if line.endswith('{') or line.endswith(':'):
                    nesting_level += 1

        # Check for breaks and continues
        for keyword in break_continue_keywords:
            if re.search(rf'\b{keyword}\b', line):
                complexity += 1

        # Check for closing braces to decrease nesting level
        if line.strip() == '}':
            nesting_level = max(0, nesting_level - 1)

    return complexity