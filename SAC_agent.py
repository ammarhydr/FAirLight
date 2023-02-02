import numpy as np 
import os 
from agent import Agent
import random 
import torch
from torch import nn
# import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops
# import torch_scatter
# from collections import OrderedDict



# class Embedding_MLP(nn.Module):
#     def __init__(self, in_size, layers):
#         super(Embedding_MLP, self).__init__()
#         constructor_dict = OrderedDict()
#         for l_idx, l_size in enumerate(layers):
#             name = f"node_embedding_{l_idx}"
#             if l_idx == 0:
#                 h = nn.Linear(in_size, l_size)
#                 constructor_dict.update({name: h})
#             else:
#                 h = nn.Linear(layers[l_idx - 1], l_size)
#                 constructor_dict.update({name: h})
#             name = f"n_relu_{l_idx}"
#             constructor_dict.update({name: nn.ReLU()})

#         self.embedding_node = nn.Sequential(constructor_dict)

#     def _forward(self, x):
#         x = self.embedding_node(x)
#         return x

#     def forward(self, x, train=True):
#         if train:
#             return self._forward(x)
#         else:
#             with torch.no_grad():
#                 return self._forward(x)

# class MultiHeadAttModel(MessagePassing):
#     """
#     inputs:
#         In_agent [bacth,agents,128]
#         In_neighbor [agents, neighbor_num]
#         l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
#         d: dimension of agents's embedding
#         dv: dimension of each head
#         dout: dimension of output
#         nv: number of head (multi-head attention)
#     output:
#         -hidden state: [batch,agents,32]
#         -attention: [batch,agents,neighbor]
#     """
#     def __init__(self, d=128, dv=16, d_out=128, nv=8, suffix=-1):
#         super(MultiHeadAttModel, self).__init__(aggr='add')
#         self.d = d
#         self.dv = dv
#         self.d_out = d_out
#         self.nv = nv
#         self.suffix = suffix
#         # target is center
#         self.W_target = nn.Linear(d, dv * nv)
#         self.W_source = nn.Linear(d, dv * nv)
#         self.hidden_embedding = nn.Linear(d, dv * nv)
#         self.out = nn.Linear(dv, d_out)
#         self.att_list = []
#         self.att = None

#     def _forward(self, x, edge_index):

#         # x has shape [N, d], edge_index has shape [E, 2]
#         edge_index, _ = add_self_loops(edge_index=edge_index)
#         aggregated = self.propagate(x=x, edge_index=edge_index)  # [16, 16]
#         out = self.out(aggregated)
#         out = F.relu(out)  # [ 16, 128]
#         #self.att = torch.tensor(self.att_list)
#         return out

#     def forward(self, x, edge_index, train=True):
#         if train:
#             return self._forward(x, edge_index)
#         else:
#             with torch.no_grad():
#                 return self._forward(x, edge_index)

#     def message(self, x_i, x_j, edge_index):
#         h_target = F.relu(self.W_target(x_i))
#         h_target = h_target.view(h_target.shape[:-1][0], self.nv, self.dv)
#         agent_repr = h_target.permute(1, 0, 2)

#         h_source = F.relu(self.W_source(x_j))
#         h_source = h_source.view(h_source.shape[:-1][0], self.nv, self.dv)
#         neighbor_repr = h_source.permute(1, 0, 2)   #[nv, E, dv]
#         index = edge_index[1]  # which is target
#         e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  # [5, 64]
#         max_node = torch_scatter.scatter_max(e_i, index=index)[0]  # [5, 16]
#         max_i = max_node.index_select(1, index=index)  # [5, 64]
#         ec_i = torch.add(e_i, -max_i)
#         ecexp_i = torch.exp(ec_i)
#         norm_node = torch_scatter.scatter_sum(ecexp_i, index=index)  # [5, 16]
#         normst_node = torch.add(norm_node, 1e-12)  # [5, 16]
#         normst_i = normst_node.index_select(1, index)  # [5, 64]

#         alpha_i = ecexp_i / normst_i  # [5, 64]
#         alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)
#         alpha_i_expand = torch.permute(alpha_i_expand, (1, 2, 0))  # [5, 64, 16]
#         hidden_neighbor = F.relu(self.hidden_embedding(x_j))
#         hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)
#         hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [5, 64, 16]
#         out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)

#         self.att_list.append(alpha_i)  # [64, 16]
#         return out
#     """
#     def aggregate(self, inputs, edge_index):
#         out = inputs
#         index = edge_index[1]
#     """

#     def get_att(self):
#         if self.att is None:
#             print('invalid att')
#         return self.att


# class ColightNet(nn.Module):
#     def __init__(self, input_dimension,output_dimension):
#         super(ColightNet, self).__init__()
#         MLP_layers=[32,32]

#         self.action_space = output_dimension
#         self.modulelist = nn.ModuleList()
#         self.embedding_MLP = Embedding_MLP(input_dimension, layers=MLP_layers)
#         for i in range(2):
#             module = MultiHeadAttModel(d=MLP_layers[-1],
#                                        dv=MLP_layers[0],
#                                        d_out=MLP_layers[1],
#                                        nv=1,
#                                        suffix=i)
#             self.modulelist.append(module)
#         output_dict = OrderedDict()

#         """
#         if self.constructor_dict.get('N_LAYERS') == 0:
#             out = nn.Linear(32, self.action_space.n)
#             name = f'output'
#             output_dict.update({name: out})
#             self.output_layer = nn.Sequential(output_dict)
#         """
#         output_dict = OrderedDict()
#         if len([]) != 0:
#             # TODO: dubug this branch
#             for l_idx, l_size in enumerate(self.constructor_dict['OUTPUT_LAYERS']):
#                 name = f'output_{l_idx}'
#                 if l_idx == 0:
#                     h = nn.Linear(module.d_out, l_size)
#                 else:
#                     h = nn.Linear(self.output_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
#                 output_dict.update({name: h})
#                 name = f'relu_{l_idx}'
#                 output_dict.update({name: nn.ReLU})
#             out = nn.Linear(self.constructor_dict['OUTPUT_LAYERS'][-1], self.action_space.n)
#         else:
#             out = nn.Linear(module.d_out, self.action_space.n)
#         name = f'output'
#         output_dict.update({name: out})
#         self.output_layer = nn.Sequential(output_dict)

#     def forward(self, x, edge_index, train=True):
#         h = self.embedding_MLP.forward(x, train)
#         #TODO: implement att
#         for mdl in self.modulelist:
#             h = mdl.forward(h, edge_index, train)
#         if train:
#             h = self.output_layer(h)
#         else:
#             with torch.no_grad():
#                 h = self.output_layer(h)
#         return h
# SEED=6666
# random.seed(SEED)
# np.random.seed(SEED)
# torch.seed(SEED)




    
class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=64)
        self.layer_2 = torch.nn.Linear(in_features=64, out_features=64)
        self.output_layer = torch.nn.Linear(in_features=64, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_activation(self.output_layer(layer_2_output))
        return output

class SACAgent(Agent): 
    def __init__(self, 
        dic_agent_conf=None, 
        dic_traffic_env_conf=None, 
        dic_path=None, 
        cnt_round=None, 
        best_round=None, bar_round=None,intersection_id="0"):
        """
        #1. compute the (dynamic) static Adjacency matrix, compute for each state
        -2. #neighbors: 5 (1 itself + W,E,S,N directions)
        -3. compute len_features
        -4. self.num_actions
        """
        super(SACAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)
        
        self.num_agents=dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors=min(dic_traffic_env_conf['TOP_K_ADJACENCY'],self.num_agents)

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.memory = self.build_memory()
        
        # self.state_dim = 12 *2 + 1#* self.num_agents
        if dic_traffic_env_conf['NEIGHBOR']:
            self.state_dim = 12 * 10#* self.num_agents with neighbor
        else:
            self.state_dim = 12 + 8#* self.num_agents
        self.target_entropy = 0.98 * -np.log(1 / self.num_actions)


        if cnt_round == 0: 
            # initialization
            self.critic_local = Network(input_dimension=self.state_dim, output_dimension=self.num_actions)
            self.critic_local2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
            self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
    
            self.critic_target = Network(input_dimension=self.state_dim, output_dimension=self.num_actions)
            self.critic_target2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions)
    
            self.soft_update_target_networks(tau=1.)
    
            self.actor_local = Network(input_dimension=self.state_dim, output_dimension=self.num_actions, output_activation=torch.nn.Softmax(dim=1) )
            self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
    
            self.log_alpha = torch.tensor(np.log(1.), requires_grad=True)
            self.alpha = self.log_alpha
            self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.dic_agent_conf["LEARNING_RATE"])
        else:
            # initialization
            self.critic_local = Network(input_dimension=self.state_dim, output_dimension=self.num_actions)
            self.critic_local2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
            self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
    
            self.critic_target = Network(input_dimension=self.state_dim, output_dimension=self.num_actions)
            self.critic_target2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions)
    
            self.soft_update_target_networks(tau=1.)
    
            self.actor_local = Network(input_dimension=self.state_dim, output_dimension=self.num_actions, output_activation=torch.nn.Softmax(dim=1) )
            self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
            
            try:
                # print('init q load')
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
                # print('init q_bar load')
                self.load_network_bar("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))
        # print(self.alpha)

    def choose_action(self, count, state):
        '''
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        '''
        dic_state_feature_arrays = {} # {feature1: [inter1, inter2,..], feature2: [inter1, inter 2...]}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []

        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            # print(s[feature_name])
            if "cur_phase" in feature_name:
                dic_state_feature_arrays[feature_name].append(np.array(self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']][state[feature_name][0]]))
            else:
                dic_state_feature_arrays[feature_name].append(np.array(state[feature_name]))

        state_input = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                        self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]
        
        action_probabilities = self.get_action_probabilities(state_input)
        discrete_action = np.random.choice(range(self.num_actions), p=action_probabilities)
        return discrete_action

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(np.concatenate(state, axis=1), dtype=torch.float32)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()
    
    def reform_state(self, state):

        dic_state_feature_arrays = {} # {feature1: [inter1, inter2,..], feature2: [inter1, inter 2...]}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []


        # for s in states:
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name].append(np.array(state[feature_name]))

        state_input = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                       self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]   
        
        return np.concatenate(state_input, axis=1)

    def prepare_Xs_Y(self, memory, dic_exp_conf):

        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory
        # forget
        else:
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = memory[ind_sta: ind_end]
            print("memory size after forget:", len(memory_after_forget))

            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            sample_slice = random.sample(memory_after_forget, sample_size)
            print("memory samples number:", sample_size)

        _state = []
        _next_state = []
        _action=[]
        _reward=[]
        _constraint=[]
        for i in range(len(sample_slice)): 
            state, action, next_state, reward, constraint, _, _ ,_ = sample_slice[i]
            _state.append(self.reform_state(state))
            _next_state.append(self.reform_state(next_state))
            _action.append(action)
            _reward.append(reward)  
            _constraint.append(constraint)
            

        states_tensor = torch.tensor(np.array(_state, dtype=np.float32).squeeze(1).reshape((len(_state), -1)))
        actions_tensor = torch.tensor(_action)
        rewards_tensor = torch.tensor(_reward).float()
        next_states_tensor = torch.tensor(np.array(_next_state, dtype=np.float32).squeeze(1).reshape((len(_next_state), -1)))


        #target: [#agents,#samples,#num_actions]    
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            soft_state_values = (action_probabilities * (torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities)).sum(dim=1)

            next_q_values = rewards_tensor + self.dic_agent_conf["GAMMA"]*soft_state_values            

        soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        soft_q_values2 = self.critic_local2(states_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)            

        self.X_critic = soft_q_values
        self.X_critic2 = soft_q_values2
        
        self.Y_critic=next_q_values
        
        self.states_for_actor=states_tensor


    def train_network(self, dic_exp_conf):

        critic_square_error = torch.nn.MSELoss(reduction="none")(self.X_critic, self.Y_critic)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(self.X_critic2, self.Y_critic)

        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        critic_loss.backward()
        critic2_loss.backward()
        self.critic_optimiser.step()
        self.critic_optimiser2.step()
        
        
        actor_loss, log_action_probabilities = self.actor_loss(self.states_for_actor)
        actor_loss.backward()
        self.actor_optimiser.step()        
        
        alpha_loss = self.temperature_loss(log_action_probabilities)
        alpha_loss.backward()
        self.alpha_optimiser.step()

        self.soft_update_target_networks()
          
        
    def actor_loss(self, states_tensor):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss   
    
    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def soft_update_target_networks(self, tau=0.01):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def build_memory(self):

        return []


    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
            
        self.critic_local.load_state_dict(torch.load(os.path.join(file_path, "%s_critic.h5" % file_name)))  
        self.critic_local2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic2.h5" % file_name)))  
        
        self.actor_local.load_state_dict(torch.load(os.path.join(file_path, "%s_actor.h5" % file_name)))  
        
        self.alpha = torch.load(os.path.join(file_path, "%s_alpha.pt" % file_name))        
        self.log_alpha = self.alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.dic_agent_conf["LEARNING_RATE"])
        
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.critic_target.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target.h5" % file_name)))  
        self.critic_target2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target2.h5" % file_name)))  
        print("succeed in loading target model %s"%file_name) 

    def save_network(self, file_name):
        torch.save(self.actor_local.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_actor.h5" % file_name)) 
        torch.save(self.critic_local.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic.h5" % file_name))        
        torch.save(self.critic_local2.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic2.h5" % file_name))  
        torch.save(self.alpha, os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_alpha.pt" % file_name))  
        
        # self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def save_network_bar(self, file_name):
        torch.save(self.critic_target.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target.h5" % file_name))        
        torch.save(self.critic_target2.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target2.h5" % file_name))        

        # self.q_network_bar.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))



if __name__=='__main__':
    dic_agent_conf={
        'att_regularization': False, 
        'rularization_rate': 0.03, 
        'LEARNING_RATE': 0.001, 
        'SAMPLE_SIZE': 1000, 
        'BATCH_SIZE': 20, 
        'EPOCHS': 100, 
        'UPDATE_Q_BAR_FREQ': 5, 
        'UPDATE_Q_BAR_EVERY_C_ROUND': False, 
        'GAMMA': 0.8, 
        'MAX_MEMORY_LEN': 10000, 
        'PATIENCE': 10, 
        'D_DENSE': 20, 
        'N_LAYER': 2, 
        'EPSILON': 0.8, 
        'EPSILON_DECAY': 0.95, 
        'MIN_EPSILON': 0.2, 
        'LOSS_FUNCTION': 'mean_squared_error', 
        'SEPARATE_MEMORY': False, 
        'NORMAL_FACTOR': 20, 
        'TRAFFIC_FILE': 'sumo_1_3_300_connect_all.xml'}
    dic_traffic_env_conf={
        'ACTION_PATTERN': 'set', 
        'NUM_INTERSECTIONS': 1000, 
        'TOP_K_ADJACENCY': 1000, 
        'MIN_ACTION_TIME': 10, 
        'YELLOW_TIME': 5, 
        'ALL_RED_TIME': 0, 
        'NUM_PHASES': 2, 
        'NUM_LANES': 1, 
        'ACTION_DIM': 2, 
        'MEASURE_TIME': 10, 
        'IF_GUI': False, 
        'DEBUG': False, 
        'INTERVAL': 1, 
        'THREADNUM': 8, 
        'SAVEREPLAY': True, 
        'RLTRAFFICLIGHT': True, 
        'DIC_FEATURE_DIM': {'D_LANE_QUEUE_LENGTH': (4,), 'D_LANE_NUM_VEHICLE': (4,), 'D_COMING_VEHICLE': (4,), 'D_LEAVING_VEHICLE': (4,), 'D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1': (4,), 'D_CUR_PHASE': (8,), 'D_NEXT_PHASE': (8,), 'D_TIME_THIS_PHASE': (1,), 'D_TERMINAL': (1,), 'D_LANE_SUM_WAITING_TIME': (4,), 'D_VEHICLE_POSITION_IMG': (4, 60), 'D_VEHICLE_SPEED_IMG': (4, 60), 'D_VEHICLE_WAITING_TIME_IMG': (4, 60), 'D_PRESSURE': (1,), 'D_ADJACENCY_MATRIX': (3,)}, 
        'LIST_STATE_FEATURE': ['cur_phase', 'lane_num_vehicle', 'adjacency_matrix'], 
        'DIC_REWARD_INFO': {'flickering': 0, 'sum_lane_queue_length': 0, 'sum_lane_wait_time': 0, 'sum_lane_num_vehicle_left': 0, 'sum_duration_vehicle_left': 0, 'sum_num_vehicle_been_stopped_thres01': 0, 'sum_num_vehicle_been_stopped_thres1': 0, 'pressure': -0.25}, 
        'LANE_NUM': {'LEFT': 1, 'RIGHT': 1, 'STRAIGHT': 1}, 
        'PHASE': {'sumo': {0: [0, 1, 0, 1, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 1, 0, 1]}, 'anon': {1: [0, 1, 0, 1, 0, 0, 0, 0], 2: [0, 0,0, 0, 0, 1, 0, 1], 3: [1, 0, 1, 0, 0, 0, 0, 0], 4: [0, 0, 0, 0, 1, 0, 1, 0]}}, 
        'ONE_MODEL': False, 
        'NUM_AGENTS': 1, 
        'SIMULATOR_TYPE': 'sumo', 
        'BINARY_PHASE_EXPANSION': True, 
        'NUM_ROW': 3, 
        'NUM_COL': 1, 
        'TRAFFIC_FILE': 'sumo_1_3_300_connect_all.xml', 
        'ROADNET_FILE': 'roadnet_1_3.json'}
    dic_path={
        'PATH_TO_MODEL': 'model/0106_afternoon_1x3_300_GCN_time_test/sumo_1_3_300_connect_all.xml_01_06_17_23_51', 
        'PATH_TO_WORK_DIRECTORY': 'records/0106_afternoon_1x3_300_GCN_time_test/sumo_1_3_300_connect_all.xml_01_06_17_23_51', 
        'PATH_TO_DATA': 'data/template_lsr/1_3', 
        'PATH_TO_PRETRAIN_MODEL': 'model/initial/sumo_1_3_300_connect_all.xml', 
        'PATH_TO_PRETRAIN_WORK_DIRECTORY':'records/initial/sumo_1_3_300_connect_all.xml', 
        'PATH_TO_PRETRAIN_DATA': 'data/template', 
        'PATH_TO_AGGREGATE_SAMPLES': 'records/initial', 
        'PATH_TO_ERROR': 'errors/0106_afternoon_1x3_300_GCN_time_test'}
    cnt_round=200
    one_agent=SACAgent(
        dic_agent_conf=dic_agent_conf, 
        dic_traffic_env_conf=dic_traffic_env_conf, 
        dic_path=dic_path, 
        cnt_round=cnt_round,        
    )
    one_model=one_agent.build_network()
    one_agent.build_network_from_copy(one_model)



