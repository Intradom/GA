"""

Symbolic regression using tree structure approach

Valid operators: +, -, *, / 

Valid operands (suppose x is an input variable): const, x, cos(x), sin(x)

"""

class Node(object):
    def __init__(self, type, data):
        self.type = type
        self.data = data
        self.parent
        self.child_left = None
        self.child_right = None

    def getType(self):
        return self.type
        
    def getData(self):
        return self.data

    def getLeftChild(self):
        return self.child_left
        
    def getLeftChild(self):
        return self.child_left

    def setData(self,newdata):
        self.data = newdata

    def setNext(self,newnext):
        self.next = newnext
        
def main():
    

if __name__ == "__main__":
  main()