"""

Symbolic regression using instruction set approach

Valid operators: +, -, *, / 

Valid operands (suppose x is an input variable): const, x, cos(x), sin(x)

"""

import numpy as np

POSSIBLE_TYPES = ["load", "cos", "sin", "add", "sub", "mul", "div"]
BI_OP_START = 3

CONST_MAX = 10 # Loaded constants will range from -CONST_MAX to CONST_MAX
INITIAL_INSTRUCTIONS = 10

class Instruction(object):
    def __init__(self, i_num, var):
        if var:
            self.var = True # Will never mutate these instructions
            self.type = "load"
            self.op1 = i_num # Later dictates what variable num it indexed into variables array
            self.op2 = None
        else:
            self.var = False
            self.mutate(i_num)

    def mutate(self, i_num):
        r = 0
    
        if i_num == 0:
            self.type = "load"
        else:
            r = np.random.randint(len(POSSIBLE_TYPES))
            self.type = POSSIBLE_TYPES[r]
        
        if self.type == "load":
            self.op1 = np.float16((np.random.rand() - 0.5) * 2 * CONST_MAX)
            self.op2 = None
        else:              
            if r < BI_OP_START: # Requires only 1 operand
                self.op1 = np.random.randint(i_num)
                self.op2 = None
            else: # Requires two operands
                self.op1 = np.random.randint(i_num)
                self.op2 = np.random.randint(i_num)

    def setData(self, newdata):
        self.data = newdata
        
    def print_instruction(self, num):
        if self.var:
            print(str(num) + ". " + self.type + ": " + "X_" + str(self.op1) + ", " + str(self.op2))
        else:
            print(str(num) + ". " + self.type + ": " + str(self.op1) + ", " + str(self.op2))

# For debugging
def print_instuctions(num_vars, instructions):
    for i in range(instructions.size):
        instructions[i].print_instruction(i)

def main():
    # Initialize
    num_vars = 5
    results = np.empty([0, 1])
    instructions = np.empty([0, 1])

    for i in range(num_vars):
        instructions = np.append(instructions, Instruction(i, True))
    
    for i in range(INITIAL_INSTRUCTIONS):
        instructions = np.append(instructions, Instruction(i + num_vars, False))
        
    print_instuctions(num_vars, instructions)

    # Evaluate fitness (RMSE)

    """ Loop """
    # Mutate

    # Evaluate fitness (RMSE)

    # If pass stopping criteria, exit loop. Else (end regardless of result if time exceeded), revert if mutation made worse and loop; otherwise, keep changes and loop again

if __name__ == "__main__":
    main()
