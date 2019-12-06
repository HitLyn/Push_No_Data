import numpy as np
import os
import sys
import gym

class Predictor():
    def __init__(self, env, c_o, c_g, c_d, time_steps):
        self.env = gym.make('FetchPush-v5')
        # self.env = env
        self.env.reset()
        self.T = time_steps # time_steps to predict

        self.trajectory_length = self.env._max_episode_steps
        self.input_sequence_len = 10

        # self.obstacle_position = np.zeros([2]) # obstacle absolute position
        self.goal_position = np.zeros([3]) # goal_position and orientation absolute data(relative to world coordinate)

        self.c_o = c_o # cost coefficient for obstacle distance
        self.c_g = c_g # cost coefficient for goal distance
        self.c_d = c_d # cost coefficient for distance between object and robot


    def catch_up(self, goal_state, initial_state, action_history, step):
        """create a new env and catch up the current state by act the same as act history
        Args:
            goal_state,
            initial_state,
            action_history,
            step: (int) timestep
        """
        # set initial state including  object goal and initial positions
        self.env.state_reset(goal_state, initial_state)
        # self.obstacle_position = self.env.unwrapped.sim.data.get_site_xpos('obstacle')[:2]
        goal_position = self.env.unwrapped._get_obs()['desired_goal']
        self.goal_position = np.array([goal_position[0], goal_position[1], 2*np.arccos(goal_position[3])])
        for step in range(len(action_history)):
            self.env.step(action_history[step])

        # code to make sure the env catches up actually


    def predict(self, action, step):
        assert action.shape == (2,)
        act = np.zeros([4])
        act[:2] = action[:]

        obs, r, _, _ = self.env.step(act) # [delta_x, delta_y, delta_theta]
        observation = obs['observation']

        # compute the cost
        cost = self.cost_fun(observation[6:9], observation[:2], self.goal_position)

        return cost

    def cost_fun(self, object_position, robot_position, goal_position):
        c_o = self.c_o
        c_g = self.c_g
        c_d = self.c_d
        object_position_ = object_position[:2]
        object_rotation_ = object_position[2]

        # cost = c_g*np.squeeze(np.sum(np.square(object_position - goal_position))) + c_d*np.squeeze(np.sum(np.square(object_position_ - robot_position)))
        cost = c_d*np.squeeze(np.sum(np.square(object_position_ - robot_position)))
        return cost

    def _get_obs(self):
        return 0


def main():
    predictor = Predictor()

if __name__ == '__main__':
    main()
