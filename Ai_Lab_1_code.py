

#1.1
# pseudocode for a graph search agent

def graph_search_agent(problem):
# Step 1: Initialize data structures
    frontier = [] # The frontier is where we store the states to be
    explored (queue)
    explored_set = set() # Explored_set is used to keep track of
    states that have already been explored (hash table)# Add the initial state to the frontier
    frontier.append(problem.initial_state)
# Step 2: Main loop - continue until the frontier is not empty
    while frontier:
# Step 3: Pop the current state from the frontier
    current_state = frontier.pop(0) # Using pop(0) for a
    queue-like behavior
# Step 4: Check if the current state is the goal state
    if problem.is_goal_state(current_state):
# Step 5: If the goal is found, reconstruct and return
    the path
    return reconstruct_path(current_state)
# Step 6: Add the current state to the explored set
    explored_set.add(current_state)
# Step 7: Explore neighboring states
    for action in problem.actions(current_state):
    next_state = problem.result(current_state, action)
# Step 8: Check if the next state is not explored or in
    the frontier
    if next_state not in explored_set and next_state not in    frontier:
# Step 9: Add the next state to the frontier
    frontier.append(next_state)
# Step 10: If the frontier becomes empty and the goal is not
    found, return failure
return "failure"

# 1.4
# function which can backtrack and produce the path taken to reach the goal state from the source/ initial state

def backtracking_path(source_state, goal_state):
    path = [] # Initialize an empty list to store the backtracked path
    current_state = goal_state # Start from the goal state and work
    backward
    while current_state != source_state:
    action = action_taken_to_reach(current_state) # Find the action
    taken to reach the current state
    path.insert(0, action) # Prepend the action to the path (insert
    at the beginning)
    current_state = parent_of(current_state) # Move to the parent
    state of the current state
return path # Return the backtracked path

# 1.5
# Code to generate Puzzle-8 instances with the goal state at depth “d”

#  Function to generate possible actions (up, down, left, right)
def generate_actions(state):
    actions = []
    empty_index = state.index('_')
    row, col = divmod(empty_index, 3)
    if row > 0:
    actions.append('Up')
    if row < 2:
    actions.append('Down')
    if col > 0:
    actions.append('Left')
    if col < 2:
    actions.append('Right')
    return actions

# Function to perform an action and generate the next state
def apply_action(state, action):
    new_state = list(state)
    empty_index = new_state.index('_')
    if action == 'Up':
    new_state[empty_index], new_state[empty_index - 3] =
    new_state[empty_index - 3], new_state[empty_index]
    elif action == 'Down':
    new_state[empty_index], new_state[empty_index + 3] =
    new_state[empty_index + 3], new_state[empty_index]
    elif action == 'Left':
    new_state[empty_index], new_state[empty_index - 1] =
    new_state[empty_index - 1], new_state[empty_index]
    elif action == 'Right':
    new_state[empty_index], new_state[empty_index + 1] =
    new_state[empty_index + 1], new_state[empty_index]
    return tuple(new_state)
    initial_state = ('_',1, 2, 3, 4,5, 7, 8, 6)
    goal_state = (1, 2, 3, 4, 5, 6, 7, 8, '_')
    max_depth = 16
    solution_path, frontier, explored_set, parents, depth_time_memory =
    graph_search_agent(initial_state, goal_state, max_depth)
    if solution_path:
    visualize_solution(initial_state, solution_path)
    else:
    print("No solution found.")

# 1.6
# Function to get the current process's memory usage

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss
    print("Depth | Time Elapsed (s) | Memory Used (bytes)")
    print("------|------------------|-------------------")
    for depth, time_elapsed, memory_used in depth_time_memory:
    print(f"{depth:<6}|{time_elapsed:<18}|{memory_used}")

