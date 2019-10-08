from queue import PriorityQueue
from puzzle import Puzzle
from time import time


def Astar_search(initial_state):
    count = 0
    explored = []
    start_node = Puzzle(initial_state, None, None, 0, True)
    print("Start State:", start_node)
    q = PriorityQueue()
    q.put((start_node.evaluation_function, count, start_node))

    while not q.empty():
        node = q.get()
        node = node[2]
        explored.append(node.state)
        if node.goal_test():
            return node.find_solution()
        children = node.generate_child()
        print("Successors: \n")
        for child in children:
            print(child)
            if child.state not in explored:
                count += 1
                q.put((child.evaluation_function, count, child))
                print("f(n)", child.evaluation_function)
    return


if __name__ == "__main__":
    state = [1, 3, 4,
             8, 6, 2,
             7, 0, 5]
    print("Start state", state)
    astar = Astar_search(state)
    t0 = time()
    t1 = time() - t0
    print('A*:', astar)
    print('time:', t1)
    print()
