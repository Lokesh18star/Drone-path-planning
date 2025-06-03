# import numpy as np
# import tensorflow as tf
# import streamlit as st
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout

# # Set up Streamlit
# st.title("Drone Path Planning")

# # Initialize environment, model, and planner
# class DroneEnvironment:
#     def __init__(self, size=(50, 50, 30), num_obstacles=5):
#         self.size = size
#         self.obstacles = self._generate_obstacles(num_obstacles)
#         self.start = np.array([0, 0, 0])
#         self.goal = np.array([size[0]-1, size[1]-1, size[2]-1])
        
#     def _generate_obstacles(self, num_obstacles):
#         obstacles = []
#         for _ in range(num_obstacles):
#             obstacle = np.array([np.random.randint(0, self.size[0]),
#                                  np.random.randint(0, self.size[1]),
#                                  np.random.randint(0, self.size[2])])
#             obstacles.append(obstacle)
#         return np.array(obstacles)
    
#     def is_collision(self, position):
#         for obstacle in self.obstacles:
#             if np.all(np.abs(position - obstacle) < 3):
#                 return True
#         return False
    
#     def get_state(self, position):
#         normalized_pos = position / np.array(self.size)
#         normalized_goal = self.goal / np.array(self.size)
#         obstacle_distances = np.linalg.norm(self.obstacles - position, axis=1)
#         min_obstacle_dist = np.min(obstacle_distances) / np.max(self.size)
#         return np.concatenate([normalized_pos, normalized_goal, [min_obstacle_dist]])

# class PathPlanningModel:
#     def __init__(self, input_dim=7, action_dim=3):
#         self.model = self._build_model(input_dim, action_dim)
        
#     def _build_model(self, input_dim, action_dim):
#         model = Sequential([
#             Dense(64, activation='relu', input_dim=input_dim),
#             Dropout(0.1),
#             Dense(128, activation='relu'),
#             Dropout(0.1),
#             Dense(64, activation='relu'),
#             Dense(action_dim, activation='tanh')
#         ])
#         model.compile(optimizer='adagrad', loss='mse')
#         return model
    
#     def predict_action(self, state):
#         return self.model.predict(state.reshape(1, -1))[0]
    
#     def train(self, states, actions, epochs=30, batch_size=32):
#         return self.model.fit(states, actions, epochs=epochs, 
#                             batch_size=batch_size, verbose=1)

# class DronePathPlanner:
#     def __init__(self, environment):
#         self.env = environment
#         self.model = PathPlanningModel()
        
#     def generate_training_data(self, num_episodes=100, steps_per_episode=50):
#         states = []
#         actions = []
        
#         for episode in range(num_episodes):
#             current_pos = self.env.start.copy()
            
#             for _ in range(steps_per_episode):
#                 state = self.env.get_state(current_pos)
#                 direction = self.env.goal - current_pos
#                 action = direction / np.linalg.norm(direction)
#                 action += np.random.normal(0, 0.1, size=3)
#                 action = np.clip(action, -1, 1)
                
#                 states.append(state)
#                 actions.append(action)
                
#                 new_pos = current_pos + action * 3
#                 if not self.env.is_collision(new_pos):
#                     current_pos = new_pos
                
#                 if np.all(np.abs(current_pos - self.env.goal) < 3):
#                     break
                    
#             if episode % 10 == 0:
#                 print(f"Generated episode {episode}/{num_episodes}")
                    
#         return np.array(states), np.array(actions)
    
#     def train(self, num_episodes=100):
#         print("Generating training data...")
#         states, actions = self.generate_training_data(num_episodes)
#         print("Training model...")
#         return self.model.train(states, actions)
    
#     def plan_path(self, start=None, max_steps=100):
#         if start is None:
#             current_pos = self.env.start.copy()
#         else:
#             current_pos = start.copy()
            
#         path = [current_pos.copy()]
        
#         for step in range(max_steps):
#             state = self.env.get_state(current_pos)
#             action = self.model.predict_action(state)
            
#             new_pos = current_pos + action * 3
#             if not self.env.is_collision(new_pos):
#                 current_pos = new_pos
#                 path.append(current_pos.copy())
            
#             if np.all(np.abs(current_pos - self.env.goal) < 3):
#                 break
                
#         return np.array(path)

# # Create planner instance
# env = DroneEnvironment()
# planner = DronePathPlanner(env)

# # Train the planner
# planner.train(num_episodes=100)

# # Plan and visualize a path
# path = planner.plan_path()

# # Visualize the path with Streamlit
# def visualize_path(env, path):
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Plot obstacles
#     ax.scatter(env.obstacles[:, 0], env.obstacles[:, 1], env.obstacles[:, 2], 
#               c='red', marker='s', s=100, label='Obstacles')
    
#     # Plot start and goal
#     ax.scatter(*env.start, c='green', marker='o', s=200, label='Start')
#     ax.scatter(*env.goal, c='blue', marker='*', s=200, label='Goal')
    
#     # Plot path
#     path = np.array(path)
#     ax.plot(path[:, 0], path[:, 1], path[:, 2], 'g--', linewidth=2, label='Planned Path')
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend(fontsize=10)
#     plt.title("Drone Path Planning Visualization", fontsize=14)
#     st.pyplot(fig)

# # Visualize the path
# visualize_path(env, path)

# def analyze_path(env, path):
#     path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
#     straight_line_dist = np.linalg.norm(env.goal - env.start)
#     path_efficiency = straight_line_dist / path_length * 100
    
#     min_obstacle_dist = float('inf')
#     for pos in path:
#         distances = np.linalg.norm(env.obstacles - pos, axis=1)
#         min_dist = np.min(distances)
#         min_obstacle_dist = min(min_obstacle_dist, min_dist)
    
#     st.write(f"Path Analysis:")
#     st.write(f"Total path length: {path_length:.2f} units")
#     st.write(f"Straight-line distance: {straight_line_dist:.2f} units")
#     st.write(f"Path efficiency: {path_efficiency:.2f}%")
#     st.write(f"Minimum distance to obstacles: {min_obstacle_dist:.2f} units")

# # Analyze the path
# analyze_path(env, path)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

class DroneEnvironment:
    def __init__(self, size=(50, 50, 30), num_obstacles=5):
        self.size = size
        self.obstacles = self._generate_obstacles(num_obstacles)
        self.start = np.array([0, 0, 0])
        self.goal = np.array([size[0]-1, size[1]-1, size[2]-1])
        
    def _generate_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            obstacle = np.array([np.random.randint(0, self.size[0]),
                                 np.random.randint(0, self.size[1]),
                                 np.random.randint(0, self.size[2])])
            obstacles.append(obstacle)
        return np.array(obstacles)
    
    def is_collision(self, position):
        for obstacle in self.obstacles:
            if np.all(np.abs(position - obstacle) < 3):  # Check proximity to obstacle
                return True
        return False
    
    def get_state(self, position):
        normalized_pos = position / np.array(self.size)
        normalized_goal = self.goal / np.array(self.size)
        obstacle_distances = np.linalg.norm(self.obstacles - position, axis=1)
        min_obstacle_dist = np.min(obstacle_distances) / np.max(self.size)
        return np.concatenate([normalized_pos, normalized_goal, [min_obstacle_dist]])

# Simplified Path Planner (No AI or collision avoidance, straight path from start to goal)
class SimplePathPlanner:
    def __init__(self, environment):
        self.env = environment
        
    def plan_straight_path(self):
        """Plan a straight path from start to goal"""
        path = []
        current_pos = self.env.start.copy().astype(np.float64)  # Ensure current_pos is a float
        goal_pos = self.env.goal.copy().astype(np.float64)      # Ensure goal_pos is a float
        
        # Generate a straight-line path (ignoring obstacles for now)
        path.append(current_pos.copy())
        
        # Calculate the direction from start to goal
        direction = goal_pos - current_pos
        steps = np.linalg.norm(direction) / 3  # Step size
        
        for step in range(int(steps)):
            current_pos += direction / np.linalg.norm(direction) * 3
            path.append(current_pos.copy())
            
            # If we are close enough to the goal, break
            if np.linalg.norm(goal_pos - current_pos) < 3:
                break
        
        return np.array(path)

# Streamlit app starts here
def visualize_path(env, path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles
    ax.scatter(env.obstacles[:, 0], env.obstacles[:, 1], env.obstacles[:, 2], 
               c='red', marker='s', s=100, label='Obstacles')
    
    # Plot start and goal
    ax.scatter(*env.start, c='green', marker='o', s=200, label='Start')
    ax.scatter(*env.goal, c='blue', marker='*', s=200, label='Goal')
    
    # Plot the planned path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'g--', linewidth=2, label='Planned Path')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(fontsize=10)
    plt.title("Drone Path Planning Visualization", fontsize=14)
    
    st.pyplot(fig)  # Render the plot in Streamlit

def main():
    st.title("3D Drone Path Planning")

    # Create environment and planner
    env = DroneEnvironment(size=(50, 50, 30), num_obstacles=5)
    planner = SimplePathPlanner(env)
    
    # Plan and visualize the path
    path = planner.plan_straight_path()
    
    # Display the environment and path using Streamlit
    st.subheader("Planned Path Visualization")
    visualize_path(env, path)

if __name__ == "__main__":
    main()
