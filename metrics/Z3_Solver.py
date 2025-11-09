from z3 import *


def calculate_satisfiability(z3_code):
    try:
        # Create a new solver
        s = Solver()

        # Create a new context where we'll execute the Z3 code
        loc = {}
        glob = {'Solver': Solver, 'Bool': Bool, 'Int': Int, 'Real': Real,
                'Const': Const, 'BitVec': BitVec, 'And': And, 'Or': Or,
                'Not': Not, 'Implies': Implies, 'If': If, 'solver': s}

        # Execute the Z3 code
        exec(z3_code, glob, loc)

        # Get the solver object that might have been modified in the code
        s = glob.get('solver', s)

        # Check satisfiability
        result = s.check()

        if result == sat:
            return 1
        elif result == unsat:
            return 0
        else:
            return -1

    except Exception as e:
        return -1