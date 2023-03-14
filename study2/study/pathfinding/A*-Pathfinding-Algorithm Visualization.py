import tkinter as tk
import math
import time

WIDTH = 20

root = tk.Tk()
node_frame = tk.Frame(root).grid(row = 0, column = 0)
menu_frame = tk.Frame(root).grid(row = 1, column = 0)
root.config(bg = "black")
node_styling = {"bg": "#1c1b1a", "fg": "yellow", "relief": tk.FLAT}
font = ("Helvetica", 15, "bold")

class Node:
    clicks = 0
    start_node = None
    end_node = None
    
    __slots__ = ['button','row', 'column', 'parent',
     'heuristic', 'Gcost', 'Fcost', 'start', 'end', 'barrier']
    
    def __init__(self, column, row):
        
        self.button = tk.Button(node_frame,
         width = 2,
         height = 1,
         command = lambda a=row, b=column: self.click(a, b),
         cnf = node_styling
        )
        
        self.button.grid(row = row, column = column, padx = 1, pady = 1)
        
        self.column = column
        self.row = row
        self.parent = None
        self.heuristic = 0
        self.Gcost = 0
        self.Fcost = self.Gcost + self.heuristic
        self.start = False
        self.end = False
        self.barrier = False
 
    def click(self, row, column):
        Node.clicks += 1
        if Node.clicks == 1:   
            self.button.config(bg = "green")
            self.start = True
            Node.start_node = (self.column, self.row)
        elif Node.clicks == 2:
            self.button.config(bg = "red")
            self.end = True
            Node.end_node = (self.column, self.row)
        else :
            self.button.config(bg = "black")
            self.barrier = True
    
    def get_distance(self, node_b):
        '''
        returns the distance between two nodes (10 for horizontal, 14 vertical)
        '''
        node_b = node_b
        x = abs(self.column - node_b.column)
        y = abs(self.row - node_b.row)

        if x > y:
            return (14*y + 10*(x - y))
        return (14*x + 10*(y - x))

    def neighbours(self):
        '''
        finds neighbours of the given node (current node)
        '''
        neighbours = []
        x, y = self.column, self.row
        for a in range(x - 1, x + 2):
            if a < 0 or a > 19:
                continue
            for b in range(y - 1, y + 2):
                if b < 0 or b > 19:
                    continue
                neighbours.append(node_list[a][b])
        
        neighbours.remove(node_list[x][y])
        return neighbours

    @staticmethod
    def pathfinding():
        '''
        The A* pathfinding fucntion
        '''
        OPEN = []
        CLOSED = []
        xe,ye = Node.end_node
        xs, ys = Node.start_node
        OPEN.append(node_list[xs][ys])
        while True:

            current_node = OPEN[0]
            for node in OPEN:
                if node.Fcost < current_node.Fcost or (node.Fcost == current_node.Fcost and node.heuristic < current_node.heuristic):
                    current_node = node
            OPEN.remove(current_node)
            CLOSED.append(current_node)
            
            if current_node.end == True:
                print("End node found.")
                current_node.find_path()
                return
            
            current_node.button.config(bg = "pink")
            root.update_idletasks()
            time.sleep(0.1)
            
            neighbours = current_node.neighbours()
            for neighbour in neighbours:
                if neighbour.barrier == True or neighbour in CLOSED:
                    continue
                new_movement_cost = current_node.Gcost + current_node.get_distance(neighbour)
                if new_movement_cost < neighbour.Gcost or neighbour not in OPEN:
                    neighbour.Gcost = new_movement_cost
                    neighbour.heuristic = neighbour.get_distance(node_list[xe][ye])
                    neighbour.parent = current_node
                    
                    neighbour.button.config(bg = "purple")
                    root.update_idletasks()
                    time.sleep(0.1)
                    OPEN.append(neighbour)

    def find_path(self):
        '''
        Traces a path from end node to start node
        '''
        current = self
        
        while current.start == False:
            parent = current.parent
            
            parent.button.config(bg = "grey")
            root.update_idletasks()
            time.sleep(0.1)

            current = parent

if __name__ == "__main__":    
    node_list = []
    for x in range(WIDTH):
        row_list = [Node(x, y) for y in range(WIDTH)]
        node_list.append(row_list)
    
    start_button = tk.Button(menu_frame, text = "START", 
    command = Node.pathfinding, 
    cnf = node_styling,
    font = font,
    pady = 3).grid(row = 21, column = 0, columnspan =20, sticky = tk.NSEW, pady = 1)
    
    quit_button = tk.Button(menu_frame, text = "EXIT", 
    command = root.destroy, 
    cnf = node_styling,
    font = font,
    pady = 3).grid(row = 22, column = 0, columnspan =20, sticky = tk.NSEW, pady = 1)
    
    root.tk.mainloop()