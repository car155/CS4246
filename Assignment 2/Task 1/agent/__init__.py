import gym
import gym_grid_driving
from gym_grid_driving.envs.grid_driving import LaneSpec, Point, AgentState
import itertools
import numpy as np
import os
from copy import deepcopy
import torch

class GridDrivingMDP(object):
    '''
    Class that models the crossing task as an MDP
    '''

    def __init__(self, env, lanes, width, gamma):
        super(GridDrivingMDP, self).__init__()
        self.env = env
        self.num_lanes = len(lanes)
        self.width = width
        self.gamma = gamma
        self.init_state = self.env.reset()
        self.transition_function = None
        self.reward_function = None
        self.transition_dict = {}

    def get_transition_function(self):
        '''
        Function to create the state transition dynamics of the MDP
        '''

        if self.transition_function is None :
            state_vec_dim = [self.width, self.num_lanes] * len(self.env.cars) + [2]
            action_dim = [len(self.env.actions)]
            transition_function  = np.zeros(state_vec_dim + action_dim +  state_vec_dim)
            state_action_dim_valid_assignments = [list(range(self.width)),list(range(self.num_lanes))] * len(self.env.cars) + \
                                                [[0,1]] + [list(range(len(self.env.actions)))]
            current_state_list = itertools.product(*state_action_dim_valid_assignments)
            for cur_state in list(current_state_list) :
                self.populate_transition_function(cur_state[:-1], cur_state[-1], transition_function)      
            self.transition_function = transition_function
        return self.transition_function

    def populate_transition_function(self, state, action, function_matrix) :
        '''
        Function to populate the next state probabilities given the current state and action 
        '''

        #  DONE VARIABLE 
        if state[-1] == 1:
            transition_function_index = tuple(np.array([*state]+[action]+[*state]).T)
            function_matrix[transition_function_index] = 1.
        else:
            position_range_list = []
            for index in range(len(self.env.cars)) :
                car_x = state[2*index]
                car_y = state[2*index+1]
                if index == 0 : #Agent
                    if action == 0 : # UP
                        position_range_list += [[max(car_x-1, 0)], [max(car_y-1, 0)]]
                    elif action == 1 : # Down
                        position_range_list += [[max(car_x-1, 0)], [min(car_y+1, self.num_lanes -1)]]
                    else : # forward actions
                        x_shift = (action-2) + self.env.cars[0].speed_range[0]
                        position_range_list += [[max(car_x+x_shift, 0)], [car_y]]
                else : #Cars

                    position_range_list += [list((np.array(range(*tuple(self.env.cars[index].speed_range + np.array([0,1])))) + car_x) % self.width), [car_y]]
            position_range_list += [[0,1]]
            possible_next_states = itertools.product(*position_range_list)

            p = self.env.p
            average_speed = [0]
            for car_index in range(1, len(self.env.cars)) :
                average_speed += [np.round(np.average(self.env.cars[car_index].speed_range))]

            self.transition_dict[state,action] = []
            for next_state in possible_next_states :
                isNextStateGoal = (next_state[0] == self.init_state.finish_position.x) and (next_state[1] == self.init_state.finish_position.y)
                isNextStateFinal = (next_state[0] == 0)
                if next_state[-1]== 0 and (self.collision(state,next_state) or isNextStateGoal or isNextStateFinal):
                    continue
                if next_state[-1] == 1 and not (self.collision(state, next_state) or isNextStateGoal or isNextStateFinal) :
                    continue
                else :
                    probability = 1.0
                    for index in range(1, len(self.env.cars)) :
                        if int((state[2*index] + average_speed[index]) % self.width) == next_state[2*index] :
                            probability = probability * ((1-p) + float(p)/len(position_range_list[2*index]))
                        else :
                            probability = probability * (float(p)/len(position_range_list[2*index]))
                    transition_function_index = tuple(np.array([*state] + [action] + [*next_state]).T)
                    function_matrix[transition_function_index] = probability
                    self.transition_dict[state,action] += [next_state, probability]

    def collision(self, state, next_state):
        '''
        Function to detect collision between the agent and a car in the grid world
        '''

        blocked_cells = []
        for index in range(1,len(self.env.cars)):
            x_current, x_next = state[2*index], next_state[2*index]
            if x_current > x_next:
                car_trail = list(range(x_next, x_current))
            else:
                car_trail = list(set(range(self.width)) - set(range(state[2*index], next_state[2*index])))
            
            blocked_cells += list(itertools.product(car_trail,[state[2*index+1]]))

        x_current, y_current = state[0], state[1]
        x_next, y_next = next_state[0], next_state[1]
        if y_current == y_next:
            agent_trail = list(range(x_next, x_current))
            agent_cells = itertools.product(agent_trail, [y_current])
        elif y_current > y_next:##UP
            agent_cells = [(max(x_current-1,0),y_current),(max(x_current-1,0), max(y_next, 0))]
        else:##DOWN
            agent_cells = [(max(x_current-1,0),y_current),(max(x_current-1,0), min(y_next, self.num_lanes-1))]

        intersection_cells = [agent_cell for agent_cell in agent_cells for blocked_cell in blocked_cells if agent_cell==blocked_cell]

        if intersection_cells == []:
            return False 
        else:
            return True


    def get_reward_function(self) :
        '''
        Function to populate the reward function of the gym_grid_driving MDP. 
        '''

        if self.reward_function is None :
            state_vec_dim = [self.width, self.num_lanes] * len(self.env.cars) + [2]
            reward_function  = np.zeros(state_vec_dim + state_vec_dim)
            state_dim_valid_assignments = [list(range(self.width)),list(range(self.num_lanes))] * len(self.env.cars) + [[0,1]]
            current_state_list = itertools.product(*state_dim_valid_assignments)
            next_state_list = itertools.product(*state_dim_valid_assignments)
            fin_pos_x = self.init_state.finish_position.x
            fin_pos_y = self.init_state.finish_position.y
            terminal_states = [state for state in current_state_list if state[-1] == 0]
            final_states = [state for state in next_state_list if ((state[0] == fin_pos_x) and (state[1] == fin_pos_y) and (state[-1] == 1))]
            
            for cur_state, next_state in itertools.product(terminal_states, final_states):
                reward_function_index = tuple(np.array([*cur_state] + [*next_state]).T)
                reward_function[reward_function_index] = 1.

            self.reward_function = reward_function
        return self.reward_function

def matmul(inp_a, inp_b):
    '''
    Use this function for matrix multiplication
    param:
    inp_a : numpy array 
    inp_b : numpy array 
    '''
    inp_a = torch.from_numpy(inp_a)
    inp_b = torch.from_numpy(inp_b)
    out = torch.matmul(inp_a, inp_b).numpy()
    return out

def value_iteration(trn_fn, rwd_fn, gamma):
    '''
    FILL ME : Complete with Value Iteration routine to 
              return value function and policy function
    param: 
    trn_fn : shape - (|A|x|S|x|S|), trn_fn[a,s,s'] = P(s'|s,a)
    rwd_fn : shape - (|S|x|S|), rwd_fn[s,s'] = Reward of moving from s to s'
    gamma : Discount factor of MDP
    
    Returns:
    value_fn : shape - (|S|), the optimal value function of the MDP
    policy : shape - (|S|),  the optimal policy of the MDP
    '''
    value_fn = np.zeros(rwd_fn.shape[0], dtype=float)    
    policy = np.zeros(rwd_fn.shape[0], dtype=int)

    delta = 0.001                        
    '''
    Fill up the code for Value Iteration here

    Value iteration should iterate until 
    |V_{t}(s) - V_{t+1}(s)| < delta for all states. We set delta = 0.001 

    Caution : Do NOT use np.multiply()/np.dot()/np.matmul() to perform 
              matrix multiplications as is interferes with the evaluation script.
              Instead you can use "matmul()" function defined above
    '''

    value_fn_prime = np.zeros(rwd_fn.shape[0])
    size_s = value_fn.size

    while True:
        value_fn = value_fn_prime; sigma = 0
        
        for s in range(0, size_s):
            value_fn_prime[s], policy[s] = bellmanUpdate(s, trn_fn, rwd_fn, value_fn, gamma)

        sigma = value_fn_prime - value_fn

        if sigma.max() < delta:
            return value_fn, policy


''' HELPER FUNCTION '''
def bellmanUpdate(s, trn_fn, rwd_fn, value_fn, gamma):
    p = trn_fn[:, s, :]
    r = rwd_fn[s, :]
    
    # U(s) = max(sum(P(s'|s,a)R(s,s') + gamma*P(s'|s,a)U(s'))) 
    #      = max(sum(P(s'|s,a)R(s,s')) + gamma*sum(P(s'|s,a)U(s')))
    expected_value_of_action = matmul(p, r) + gamma * matmul(p, value_fn)
    return expected_value_of_action.max(), np.argmax(expected_value_of_action) # return best utility and action

def extractValueAndPolicy(env, lanes, width, gamma):
    '''
    Function that constructs the MDP, solves it with value iteration, and returns the optimal policy
    '''
    mdp = GridDrivingMDP(env=env, lanes=lanes, width=width,gamma=gamma)
    tran = mdp.get_transition_function()
    rwd = mdp.get_reward_function()

    rwds = rwd.reshape(2 * (len(lanes) * width) ** len(env.cars),2 * (len(lanes) * width) ** len(env.cars))
    trans = tran.reshape(2 * (len(lanes) * width) ** len(env.cars), 5, 2 * (len(lanes) * width) ** len(env.cars))
    trans = trans.transpose(1,0,2)

    ### Value Iteration 
    value_function, policy_function = value_iteration(trn_fn=trans, rwd_fn=rwds, gamma=gamma)
    pol = policy_function.reshape([width,len(lanes)]*len(env.cars) + [2])
    val = value_function.reshape([width,len(lanes)]*len(env.cars) + [2])
    return pol


def getStateTuple(_state):
    '''
    Helper function to convert an env state to a state feature vector.
    '''

    cars = [_state.agent] + list(_state.cars)
    done = _state.agent_state != AgentState.alive
    state = []
    for car in cars:
        state += [car.position.x, car.position.y]
    state.append(int(done==True))
    return state


try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass

class VIAgent(Agent):
    def initialize(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.policy = extractValueAndPolicy(self.env, self.env.lanes, self.env.width, self.gamma)
    
    def step(self, state, *args, **kwargs) :
        state_tuple = getStateTuple(state)
        action = self.policy[tuple(state_tuple)]
        return self.env.actions[action]
    
def create_agent(test_case_env, *args, **kwargs):
    return VIAgent()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('testcase', type=int, help='test case number')
    args = parser.parse_args()

    ### Sample test cases. 
    test_config = [{'lanes' : [LaneSpec(0, [-2, -1])] *5,'width' :9, 'gamma' : 0.9, 'seed' : 15, 'fin_pos' : Point(0,0), 'agent_pos': Point(8,4),'stochasticity': 1.  },
                   {'lanes' : [LaneSpec(1, [-2, -1])] *2,'width' :4, 'gamma' : 0.9, 'seed' : 15, 'fin_pos' : Point(0,1), 'agent_pos': Point(3,1),'stochasticity': 1.  },
                   {'lanes' : [LaneSpec(1, [-3, -1])] *2 + [LaneSpec(0, [0, 0])],'width' :4, 'gamma' : 0.9, 'seed' : 100, 'fin_pos' : Point(0,0), 'agent_pos': Point(3,2),'stochasticity': .5 },
                   {'lanes' : [LaneSpec(0, [0, 0])] + [LaneSpec(1, [-3, -1])] *2,'width' :4, 'gamma' : 0.5, 'seed' : 128, 'fin_pos' : Point(0,0), 'agent_pos': Point(3,2),'stochasticity': 0.75 },
                   {'lanes' : [LaneSpec(1, [-3, -1])] *2 + [LaneSpec(0, [0, 0])],'width' :4, 'gamma' : 0.99, 'seed' : 111, 'fin_pos' : Point(0,0), 'agent_pos': Point(3,2),'stochasticity': .5 },
                   {'lanes' : [LaneSpec(1, [-3, -1]), LaneSpec(0, [0, 0]), LaneSpec(1, [-3, -1])] ,'width' :4, 'gamma' : 0.999, 'seed' : 125, 'fin_pos' : Point(0,0), 'agent_pos': Point(3,2),'stochasticity': 0.9 }]
                   
    test_case_number = args.testcase
    LANES = test_config[test_case_number]['lanes']
    WIDTH = test_config[test_case_number]['width']
    RANDOM_SEED = test_config[test_case_number]['seed']
    GAMMA = test_config[test_case_number]['gamma']
    FIN_POS = test_config[test_case_number]['fin_pos']
    AGENT_POS = test_config[test_case_number]['agent_pos']
    stochasticity = test_config[test_case_number]['stochasticity']
    env = gym.make('GridDriving-v0', lanes=LANES, width=WIDTH, 
                   agent_speed_range=(-3,-1), finish_position=FIN_POS, agent_pos_init=AGENT_POS,
                   stochasticity=stochasticity, tensor_state=False, flicker_rate=0., mask=None, random_seed=RANDOM_SEED)
    actions = env.actions
    env.render()
     
    pol = extractValueAndPolicy(deepcopy(env), LANES, WIDTH, GAMMA)
    
    state = env.reset()
    done = False
    while not done:
        state = getStateTuple(state)
        action = pol[tuple(state)]
        print (env.actions[action])
        state, reward, done, info = env.step(env.actions[action])
        env.render()