import cv2
import numpy as np
from queue import PriorityQueue
from datetime import datetime
from PIL import Image, ImageTk
import tkinter as tk
# from tkinter import filedialog


class MazeSolver:
    def __init__(self, maze_path):
        self.maze_path = maze_path
        self.maze = None
        self.start_pixel = None
        self.end_pixel = None
        self.wall_color = (0, 0, 0)
        self.start_color = (255, 0, 0)
        self.end_color = (0, 0, 255)
        self.visited_color = (0, 255, 0)
        self.failed_color = (255, 0, 0)
        self.path_color = (0, 0, 255)
        self.path_thickness = 5
        self.show_path = False
        self.show_visited = False
        self.show_failed = False
        self.show_path_window = None
        self.load_image()

    def load_image(self):
        self.maze = cv2.imread(self.maze_path)
        self.maze = cv2.cvtColor(self.maze, cv2.COLOR_BGR2RGB)
        self.maze = np.asarray(self.maze)
        self.start_pixel = np.argwhere(np.all(self.maze == self.start_color, axis=-1))[0]
        self.end_pixel = np.argwhere(np.all(self.maze == self.end_color, axis=-1))[0]

    def get_neighbors(self, pixel):
        row, col = pixel
        neighbors = []
        # Check top neighbor
        if row > 0 and not np.array_equal(self.maze[row - 1][col], self.wall_color):
            neighbors.append((row - 1, col))
        # Check bottom neighbor
        if row < self.maze.shape[0] - 1 and not np.array_equal(self.maze[row + 1][col], self.wall_color):
            neighbors.append((row + 1, col))
        # Check left neighbor
        if col > 0 and not np.array_equal(self.maze[row][col - 1], self.wall_color):
            neighbors.append((row, col - 1))
        # Check right neighbor
        if col < self.maze.shape[1] - 1 and not np.array_equal(self.maze[row][col + 1], self.wall_color):
            neighbors.append((row, col + 1))
        return neighbors

    def heuristic(self, a, b):
        # Calculate the Manhattan distance between pixels a and b
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_search(self):
        start = tuple(self.start_pixel)
        end = self.end_pixel

        visited = {}
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                visited[(i, j)] = False

        came_from = {}
        cost_so_far = {}
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                came_from[(i, j)] = None
                cost_so_far[(i, j)] = np.inf
        came_from[start] = start
        cost_so_far[start] = 0

        frontier = PriorityQueue()
        frontier.put(start, 0)


        while not frontier.empty():
            current = frontier.get()
            if np.array_equal(current, end):
                break

            for _next in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if new_cost < cost_so_far[_next]:
                    cost_so_far[_next] = new_cost
                    priority = new_cost + self.heuristic(end, _next)
                    frontier.put(_next, priority)
                    came_from[_next] = current
                    visited[_next] = True

        # Backtrack the path from end to start
        path = []
        current = end
        while not np.all(current == start):
            path.append(current)
            current = came_from[tuple(current)]
        path.append(start)
        path.reverse()

        # Color the path in the maze image
        if self.show_path:
            for i in range(len(path) - 1):
                cv2.line(self.maze, path[i][::-1], path[i + 1][::-1], self.path_color, self.path_thickness)

        # Show visited pixels in the maze image
        if self.show_visited:
            for pixel in visited:
                if visited[pixel]:
                    cv2.circle(self.maze, pixel[::-1], 1, self.visited_color, -1)

        # Show failed paths in the maze image
        if self.show_failed:
            for pixel in came_from:
                if came_from[pixel] is not None and not visited[pixel]:
                    cv2.line(self.maze, pixel[::-1], came_from[pixel][::-1], self.failed_color, self.path_thickness)

        return path


class MazeSolverGUI:
    def __init__(self, maze_solver):
        self.maze_solver = maze_solver

        self.root = tk.Tk()
        self.root.title("Maze Solver")

        self.canvas_width = self.maze_solver.maze.shape[1]
        self.canvas_height = self.maze_solver.maze.shape[0]
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.image = ImageTk.PhotoImage(Image.fromarray(self.maze_solver.maze))
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)

        self.show_path_var = tk.BooleanVar()
        self.show_visited_var = tk.BooleanVar()
        self.show_failed_var = tk.BooleanVar()

        self.show_path_cb = tk.Checkbutton(self.root, text="Show Path", variable=self.show_path_var,
                                           command=self.update_display)
        self.show_visited_cb = tk.Checkbutton(self.root, text="Show Visited", variable=self.show_visited_var,
                                              command=self.update_display)
        self.show_failed_cb = tk.Checkbutton(self.root, text="Show Failed", variable=self.show_failed_var,
                                             command=self.update_display)
        self.show_path_cb.pack()
        self.show_visited_cb.pack()
        self.show_failed_cb.pack()

        self.solve_button = tk.Button(self.root, text="Solve", command=self.solve_maze)
        self.solve_button.pack()

        self.path_window = None

        self.root.mainloop()

    def update_display(self):
        self.maze_solver.show_path = self.show_path_var.get()
        self.maze_solver.show_visited = self.show_visited_var.get()
        self.maze_solver.show_failed = self.show_failed_var.get()

        new_maze = self.maze_solver.maze.copy()

        if self.maze_solver.show_path:
            path = self.maze_solver.get_path()
            for pixel in path:
                new_maze[pixel[0], pixel[1]] = self.maze_solver.path_color

        if self.maze_solver.show_visited:
            visited = self.maze_solver.get_visited()
            for pixel in visited:
                if pixel not in path:
                    new_maze[pixel[0], pixel[1]] = self.maze_solver.visited_color

        if self.maze_solver.show_failed:
            failed = self.maze_solver.get_failed()
            for pixel in failed:
                if pixel not in path and pixel not in visited:
                    new_maze[pixel[0], pixel[1]] = self.maze_solver.failed_color

        self.image = ImageTk.PhotoImage(Image.fromarray(new_maze))
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)

    def solve_maze(self):
        start_time = datetime.now()
        self.maze_solver.a_star_search()
        end_time = datetime.now()
        print("Time taken to solve maze: ", end_time - start_time)
        self.update_display()

        if self.path_window:
            self.path_window.destroy()

        self.path_window = tk.Toplevel(self.root)
        self.path_window.title("Path")

        canvas_width = self.maze_solver.maze.shape[1]
        canvas_height = self.maze_solver.maze.shape[0]
        canvas = tk.Canvas(self.path_window, width=canvas_width, height=canvas_height)
        canvas.pack()

        path = self.maze_solver.get_path()
        for i in range(len(path) - 1):
            canvas.create_line(path[i][1], path[i][0], path[i+1][1],
                               path[i+1][0], fill=self.maze_solver.path_color, width=2)


def main():
    # Step 1: Create a MazeSolver instance
    maze_solver = MazeSolver("data/MAZES/MAZE_4.png")

    # Step 2: Find the path from start to end using A* search algorithm
    maze_solver.a_star_search()

    # Step 3: Create a MazeSolverGUI instance
    maze_gui = MazeSolverGUI(maze_solver)

    # Step 4: Show the GUI
    # maze_gui.show()


if __name__ == '__main__':
    main()
