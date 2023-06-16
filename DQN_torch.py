import numpy as np 
import os 
from agent import Agent
import random 
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


# from torch_geometric.nn import MessagePassing
# from torch_geometric.data import Data, Batch
# from torch_geometric.utils import add_self_loops
# import torch_scatter
# from collections import OrderedDict


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNNet(nn.Module):
    '''
    DQNNet consists of 3 dense layers.
    '''
    def __init__(self, input_dim, output_dim, n_opponent_actions=1):
        super(DQNNet, self).__init__()
        self.n_opponent_actions=n_opponent_actions
        self.output_dim=output_dim
        self.dense_1 = nn.Linear(input_dim, 20)
        self.dense_2 = nn.Linear(20, 20)
        self.dense_3 = nn.Linear(20, output_dim*n_opponent_actions)

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        if self.n_opponent_actions==1:
            return x
        else:
            x.view(-1, self.n_opponent_actions, self.output_dim)
        

class DQNTorch(Agent): 
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
        super(DQNTorch, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)
        
        self.num_agents=dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors=min(dic_traffic_env_conf['TOP_K_ADJACENCY'],self.num_agents)
        
        adj_ints=list(range(dic_traffic_env_conf['NUM_INTERSECTIONS']))
        adj_matrix = torch.tensor(np.eye(self.num_agents, dtype='float64')[adj_ints]).to(device)
        self.edge_index = adj_matrix.nonzero().t().contiguous()


        self.num_actions = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.memory = self.build_memory()
        self.cnt_round = cnt_round
        
        # self.state_dim = 12 *2 + 1#* self.num_agents
        if dic_traffic_env_conf['NEIGHBOR']:
            self.state_dim = 12 * 10#* self.num_agents with neighbor
        else:
            self.state_dim = 12 + 8 #* self.num_agents

        self.constrain = dic_traffic_env_conf['CONST_NUM']
        self.num_opponents=1


        if cnt_round == 0: 
            # initialization
            self.critic_local = DQNNet(self.state_dim, self.num_actions).to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
            self.critic_target = DQNNet(self.state_dim, self.num_actions).to(device)
            if self.dic_traffic_env_conf["CONSTRAINT"]:
                self.critic_local_cost = DQNNet(self.state_dim, self.num_actions).to(device)
                self.critic_optimiser_cost = torch.optim.Adam(self.critic_local_cost.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
                self.critic_target_cost = DQNNet(self.state_dim, self.num_actions).to(device)

            self.soft_update_target_networks(tau=1.)


        else:
            # initialization
            self.critic_local = DQNNet(self.state_dim, self.num_actions).to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
    
            self.critic_target = DQNNet(self.state_dim, self.num_actions).to(device)
            if self.dic_traffic_env_conf["CONSTRAINT"]:
                self.critic_local_cost = DQNNet(self.state_dim, self.num_actions).to(device)
                self.critic_optimiser_cost = torch.optim.Adam(self.critic_local_cost.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
                self.critic_target_cost = DQNNet(self.state_dim, self.num_actions).to(device)

            # try:
            self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
            self.load_network_bar("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
                # self.load_network_bar("round_{0}_inter_{1}".format(max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            # except:
                # print("fail to load network, current round: {0}".format(cnt_round))
                
        # decay the epsilon
        """
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.95,
        "MIN_EPSILON": 0.2,
        """
        if os.path.exists(
            os.path.join(
                self.dic_path["PATH_TO_MODEL"], 
                "round_-1_inter_{0}.h5".format(intersection_id))):
            #the 0-th model is pretrained model
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f'%(cnt_round,self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])
        # print(self.alpha)
        
        
    def action_att_predict(self,state, network):
        #state:[batch,agent,features and adj]
        #return:act:[batch,agent],att:[batch,layers,agent,head,neighbors]
        batch_size=len(state)
        total_features=[]
        for i in range(batch_size): 
            feature=[]
            # adj=[] 
            for j in range(self.num_agents):
                observation=[]
                for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                    if 'adjacency' in feature_name:
                        continue
                    if feature_name == "cur_phase":
                        if len(state[i][j][feature_name])==1:
                            #choose_action
                            observation.extend(self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                                        [state[i][j][feature_name][0]])
                        else:
                            observation.extend(state[i][j][feature_name])
                    elif feature_name=="lane_num_vehicle":
                        observation.extend(state[i][j][feature_name])
                feature.append(observation)
            if batch_size>1:
                feature = torch.tensor(np.reshape(np.array(feature),[self.num_agents,-1]), dtype=torch.float32).to(device)
                total_features.append(feature)                            
            else:
                feature = torch.tensor(np.reshape(np.array(feature),[self.num_agents,-1]), dtype=torch.float32).to(device)
                value = network.forward(feature)
                return value
            # total_adjs.append(adj)
            # total_features=np.reshape(np.array(total_features),[batch_size,self.num_agents,-1])
            # total_adjs=self.adjacency_index2matrix(np.array(total_adjs))  
        # batch_state = torch.stack(total_features)
        batch_state=torch.cat(total_features, dim=0)
        value = network.forward(batch_state)
        return value


    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='float64')[y]


    def adjacency_index2matrix(self,adjacency_index):
        #adjacency_index(the nearest K neighbors):[1,2,3]
        """
        if in 1*6 aterial and 
            - the 0th intersection,then the adjacency_index should be [0,1,2,3]
            - the 1st intersection, then adj [0,3,2,1]->[1,0,2,3]
            - the 2nd intersection, then adj [2,0,1,3]

        """ 
        #[batch,agents,neighbors]
        adjacency_index_new=np.sort(adjacency_index,axis=-1)
        l = self.to_categorical(adjacency_index_new,num_classes=self.num_agents)
        return l.squeeze(0).squeeze(1)

    def choose_action(self, count, state):
        '''
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        '''
        action_probabilities =self.action_att_predict([state], self.critic_local)
        
        if np.random.rand() <= self.dic_agent_conf["EPSILON"]:
            discrete_action = list(np.random.randint(0, self.num_actions, len(action_probabilities)))
        else:
            discrete_action = list(np.argmax(action_probabilities.cpu().detach().numpy(),axis=-1))
    
        return discrete_action, 1

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()
    

    def prepare_Xs_Y(self, memory, dic_exp_conf):
        
        self.memory=memory
        

    def train_network(self, dic_exp_conf):
        
        
        ind_end = len(self.memory)
        print("memory size: {0}".format(ind_end))

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = self.memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        epochs = self.dic_agent_conf["EPOCHS"]
        for i in range(epochs):        
        
            # sample the memory
            sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            sample_slice = random.sample(memory_after_forget, sample_size)
            # print("memory samples number:", sample_size)
    
            _state = []
            _next_state = []
            _action=[]
            _reward=[]
            _constraint=[]
            for i in range(len(sample_slice)):
                _state.append([])
                _next_state.append([])
                # _action.append([])
                # _reward.append([])
                # _constraint.append([])
                for j in range(self.num_agents):
                    state, action, next_state, reward, _, constraint = sample_slice[i][j]
                    _state[i].append(state)
                    _next_state[i].append(next_state)
                    _action.append(action)
                    _reward.append(reward)
                    _constraint.append(constraint)
                
    
            self.states = _state 
            self.actions = torch.tensor(_action).to(device) 
            self.rewards = torch.tensor(_reward).to(device) 
            self.cost = torch.tensor(_constraint).to(device) 
            self.next_states = _next_state 

            with torch.no_grad():
                out = self.action_att_predict(self.next_states, self.critic_target)
                target_q_values = self.rewards.reshape(-1,1) + self.dic_agent_conf["GAMMA"] * torch.max(out, dim=1)[0].reshape(-1,1)
                
                if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain:
                    out_cost = self.action_att_predict(self.next_states, self.critic_target_cost) 
                    out_cost2 = self.action_att_predict(self.next_states, self.critic_target_cost) 
                    target_q_values_cost = self.cost.reshape(-1,1) + self.dic_agent_conf["GAMMA"] * torch.max(out_cost, dim=1)[0].reshape(-1,1)
                    target_q_values_cost2 = self.cost.reshape(-1,1) + self.dic_agent_conf["GAMMA"] * torch.max(out_cost2, dim=1)[0].reshape(-1,1)
                    
                
            q_values = self.action_att_predict(self.states, self.critic_local).gather(1, self.actions.reshape(-1,1))
        
            if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain:
                q_values_cost = self.action_att_predict(self.states, self.critic_local_cost).gather(1, self.actions.reshape(-1,1))
                q_values_cost2 = self.action_att_predict(self.states, self.critic_local_cost).gather(1, self.actions.reshape(-1,1))

                min_q=torch.min(q_values, q_values_cost)
                min_target=torch.min(target_q_values, target_q_values_cost)
                
                loss = torch.nn.MSELoss(reduction="none")(min_q, min_target).mean()
                self.critic_optimiser.zero_grad()
                loss.backward()
                # clip_grad_norm_(self.critic_local.parameters(), 0.5)
                self.critic_optimiser.step()

                loss_cost = torch.nn.MSELoss(reduction="none")(q_values_cost2, target_q_values_cost2).mean()
                
                self.critic_optimiser_cost.zero_grad()
                loss_cost.backward()
                # clip_grad_norm_(self.critic_local_cost.parameters(), 0.5)
                self.critic_optimiser_cost.step()

            else:
                loss = torch.nn.MSELoss(reduction="none")(q_values, target_q_values).mean()            
                # print("Loss: ", loss)
                self.critic_optimiser.zero_grad()
                loss.backward()
                # clip_grad_norm_(self.critic_local.parameters(), 0.5)
                self.critic_optimiser.step()
            
            self.soft_update_target_networks()


    def soft_update_target_networks(self, tau=0.01):
        self.soft_update(self.critic_target, self.critic_local, tau)
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            self.soft_update(self.critic_target_cost, self.critic_local_cost, tau)

        
    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def build_memory(self):

        return []


    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
            
        self.critic_local.load_state_dict(torch.load(os.path.join(file_path, "%s_critic.h5" % file_name), map_location='cpu'))  
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            self.critic_local_cost.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_cost.h5" % file_name), map_location='cpu'))          
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.critic_target.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target.h5" % file_name), map_location='cpu'))  
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            self.critic_target_cost.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target_cost.h5" % file_name), map_location='cpu')) 
            
    def save_network(self, file_name):
        torch.save(self.critic_local.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic.h5" % file_name))        
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            torch.save(self.critic_local_cost.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_cost.h5" % file_name))  
        
    def save_network_bar(self, file_name):
        torch.save(self.critic_target.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target.h5" % file_name))        
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            torch.save(self.critic_target_cost.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target_cost.h5" % file_name)) 