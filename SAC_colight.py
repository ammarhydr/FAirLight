import numpy as np 
import os 
from agent import Agent
import random 
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
import torch_scatter
from collections import OrderedDict


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class ColightNet(torch.nn.Module):
    def __init__(self, obs_space, action_space, model_config, name):
        super(ColightNet, self).__init__()

        self.name=name
        self.num_agents=model_config['NUM_INTERSECTIONS']
        self.num_neighbors=min(model_config['TOP_K_ADJACENCY'],self.num_agents)
        
        self.num_actions = len(model_config["PHASE"][model_config['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(model_config["LANE_NUM"].values())))


        self.state_dim = obs_space #12 + 8 #* self.num_agents
        
        self.modulelist = nn.ModuleList()
        self.embedding_MLP = Embedding_MLP(self.state_dim, layers=[128, 128])
        for i in range(1):
            module = MultiHeadAttModel(d=128,dv=16,d_out=128, nv=5, suffix=1)        
            self.modulelist.append(module)
        output_dict = OrderedDict()

        """
        if self.constructor_dict.get('N_LAYERS') == 0:
            out = nn.Linear(128, self.action_space.n)
            name = f'output'
            output_dict.update({name: out})
            self.output_layer = nn.Sequential(output_dict)
        """
        output_dict = OrderedDict()
        out = nn.Linear(module.d_out, self.num_actions)
        name = f'output'
        output_dict.update({name: out})
        self.output_layer = nn.Sequential(output_dict)


    def forward(self, x, edge_index):

        h = self.embedding_MLP.forward(x)
        for mdl in self.modulelist:
            h = mdl.forward(h, edge_index)
        x = self.output_layer(h)

        if self.name=='Actor':
            return torch.nn.Softmax(dim=1)(x)
        else:
            return x
 

class Embedding_MLP(nn.Module):
    def __init__(self, in_size, layers):
        super(Embedding_MLP, self).__init__()
        constructor_dict = OrderedDict()
        for l_idx, l_size in enumerate(layers):
            name = f"node_embedding_{l_idx}"
            if l_idx == 0:
                h = nn.Linear(in_size, l_size)
                constructor_dict.update({name: h})
            else:
                h = nn.Linear(layers[l_idx - 1], l_size)
                constructor_dict.update({name: h})
            name = f"n_relu_{l_idx}"
            constructor_dict.update({name: nn.ReLU()})

        self.embedding_node = nn.Sequential(constructor_dict)

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)



    
class MultiHeadAttModel(MessagePassing):
    """
    inputs:
        In_agent [bacth,agents,128]
        In_neighbor [agents, neighbor_num]
        l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
        d: dimension of agents's embedding
        dv: dimension of each head
        dout: dimension of output
        nv: number of head (multi-head attention)
    output:
        -hidden state: [batch,agents,32]
        -attention: [batch,agents,neighbor]
    """
    def __init__(self, d=128, dv=16, d_out=128, nv=5, suffix=-1):
        super(MultiHeadAttModel, self).__init__(aggr='add')
        self.d = d
        self.dv = dv
        self.d_out = d_out
        self.nv = nv
        self.suffix = suffix
        # target is center
        self.W_target = nn.Linear(d, dv * nv)
        self.W_source = nn.Linear(d, dv * nv)
        self.hidden_embedding = nn.Linear(d, dv * nv)
        self.out = nn.Linear(dv, d_out)
        self.att_list = []
        self.att = None

    def _forward(self, x, edge_index):
        # TODO: test batch is shared or not

        # x has shape [N, d], edge_index has shape [E, 2]
        edge_index, _ = add_self_loops(edge_index=edge_index)
        aggregated = self.propagate(x=x, edge_index=edge_index)  # [16, 16]
        out = self.out(aggregated)
        out = F.relu(out)  # [ 16, 128]
        #self.att = torch.tensor(self.att_list)
        return out

    def forward(self, x, edge_index, train=True):
        if train:
            return self._forward(x, edge_index)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index)

    def message(self, x_i, x_j, edge_index):
        h_target = F.relu(self.W_target(x_i))
        h_target = h_target.view(h_target.shape[:-1][0], self.nv, self.dv)
        agent_repr = h_target.permute(1, 0, 2)

        h_source = F.relu(self.W_source(x_j))
        h_source = h_source.view(h_source.shape[:-1][0], self.nv, self.dv)
        neighbor_repr = h_source.permute(1, 0, 2)   #[nv, E, dv]
        index = edge_index[1]  # which is target
        #TODO: confirm its a vector of size E
        # method 1: e_i = torch.einsum()
        # method 2: e_i = torch.bmm()
        # method 3: e_i = (a * b).sum(-1)
        e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  # [5, 64]
        max_node = torch_scatter.scatter_max(e_i, index=index)[0]  # [5, 16]
        max_i = max_node.index_select(1, index=index)  # [5, 64]
        ec_i = torch.add(e_i, -max_i)
        ecexp_i = torch.exp(ec_i)
        norm_node = torch_scatter.scatter_sum(ecexp_i, index=index)  # [5, 16]
        normst_node = torch.add(norm_node, 1e-12)  # [5, 16]
        normst_i = normst_node.index_select(1, index)  # [5, 64]

        alpha_i = ecexp_i / normst_i  # [5, 64]
        alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)
        alpha_i_expand = torch.permute(alpha_i_expand, (1, 2, 0))  # [5, 64, 16]
        # TODO: test x_j or x_i here -> should be x_j
        hidden_neighbor = F.relu(self.hidden_embedding(x_j))
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [5, 64, 16]
        out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)

        # TODO: maybe here
        self.att_list.append(alpha_i)  # [64, 16]
        return out
    """
    def aggregate(self, inputs, edge_index):
        out = inputs
        index = edge_index[1]
    """

    def get_att(self):
        if self.att is None:
            print('invalid att')
        return self.att



class SACAgentColight(Agent): 
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
        super(SACAgentColight, self).__init__(
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
        
        
        self.state_dim = 12 + 8 #* self.num_agents
        self.target_entropy = 0.98 * -np.log(1 / self.num_actions)
        self.cost_lim = 10
        
        # # #Hangzhou
        # self.alpha_lr = 5e-06
        # self.cost_lim = -1e-5
        # self.lam_lr = 1e-07
        
        # self.alpha_lr = 3e-06
        # self.actor_lr=0.01
        # self.lam_lr = 1e-07

        self.alpha_lr = self.dic_agent_conf["ALPHA_LEARNING_RATE"]
        self.lam_lr = self.dic_agent_conf["LAM_LEARNING_RATE"]
        self.actor_lr = self.dic_agent_conf["ACTOR_LEARNING_RATE"]
        self.lam_UR = int(self.dic_agent_conf["LAM_UPDATE_RATE"])
        
        self.constrain = dic_traffic_env_conf['CONST_NUM']

        if cnt_round == 0: 
            # initialization
            self.critic_local = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            self.critic_local2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
            self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
    
            self.critic_target = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            self.critic_target2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            
            if self.dic_traffic_env_conf["CONSTRAINT"]:
                self.critic_local_cost = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
                self.critic_local_cost2= ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
                self.critic_optimiser_cost = torch.optim.Adam(self.critic_local_cost.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
                self.critic_optimiser_cost2 = torch.optim.Adam(self.critic_local_cost2.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
        
                self.critic_target_cost = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
                self.critic_target_cost2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
    
            self.soft_update_target_networks(tau=1.)
    
            self.actor_local = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Actor').to(device)
            self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)

            # self.load_network("round_{0}_inter_{1}".format(999, self.intersection_id))
            # self.load_network_bar("round_{0}_inter_{1}".format(999, self.intersection_id))

    
            self.log_alpha = torch.tensor(np.log(1.), requires_grad=True)
            self.alpha = self.log_alpha
            self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

            self.lam = torch.tensor(1.0, requires_grad=True)
            self.lam_optimiser = torch.optim.Adam([self.lam], lr=self.lam_lr)

        else:
            # initialization
            self.critic_local = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
            self.critic_local2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
            self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
    
            self.critic_target = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
            self.critic_target2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)

            if self.dic_traffic_env_conf["CONSTRAINT"]:    
                self.critic_local_cost = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
                self.critic_local_cost2= ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
                self.critic_optimiser_cost = torch.optim.Adam(self.critic_local_cost.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
                self.critic_optimiser_cost2 = torch.optim.Adam(self.critic_local_cost2.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
        
                self.critic_target_cost = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
                self.critic_target_cost2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
        
            self.actor_local =  ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Actor').to(device)
            self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)
            
            # try:
            self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
            self.load_network_bar("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
            # except:
                # print("fail to load network, current round: {0}".format(cnt_round))
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
                total_features.append(Data(x=feature, edge_index=self.edge_index))                            
            else:
                feature = torch.tensor(np.reshape(np.array(feature),[self.num_agents,-1]), dtype=torch.float32).to(device)
                value = network.forward(feature, self.edge_index)
                return value
            # total_adjs.append(adj)
            # total_features=np.reshape(np.array(total_features),[batch_size,self.num_agents,-1])
            # total_adjs=self.adjacency_index2matrix(np.array(total_adjs))  
        batch_state = Batch.from_data_list(total_features)
        value = network.forward(x=batch_state.x, edge_index=batch_state.edge_index)
        return value


    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]


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
        return l

    def choose_action(self, count, state):
        '''
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        '''
        action_probabilities=self.action_att_predict([state], self.actor_local)
        discrete_action = [np.random.choice(range(self.num_actions), p=probs) for probs in action_probabilities.cpu().detach().numpy()]
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

        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        print("memory samples number:", sample_size)

        epochs = self.dic_agent_conf["EPOCHS"]
        for k in range(epochs):   
            
            self.actor_optimiser.zero_grad()
            self.alpha_optimiser.zero_grad()
            self.critic_optimiser.zero_grad()
            self.critic_optimiser2.zero_grad()  
        
            # sample the memory
            sample_slice = random.sample(memory_after_forget, sample_size)
    
            _state = []
            _next_state = []
            _action=[]
            _reward=[]
            _constraint=[]
            for i in range(len(sample_slice)):
                _state.append([])
                _next_state.append([])
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

            # Train model
            with torch.no_grad():
                action_probabilities, log_action_probabilities = self.get_action_info(self.next_states)
                
                next_q_values_target = self.action_att_predict(self.next_states, self.critic_target) 
                next_q_values_target2 = self.action_att_predict(self.next_states, self.critic_target2)
    
                soft_state_values = (action_probabilities * (torch.min(next_q_values_target, next_q_values_target2).to(device) - self.log_alpha.exp() * log_action_probabilities)).sum(dim=1)
                next_q_values = self.rewards.reshape(-1,1) + self.dic_agent_conf["GAMMA"]*soft_state_values.reshape(-1,1)    
                
                if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain:
                    next_q_values_target_cost = self.action_att_predict(self.next_states, self.critic_target_cost)  
                    next_q_values_target_cost2 = self.action_att_predict(self.next_states, self.critic_target_cost2)
                    soft_state_values_cost = (action_probabilities * (torch.min(next_q_values_target_cost, next_q_values_target_cost2).to(device) - self.log_alpha.exp() * log_action_probabilities)).sum(dim=1)
                    next_q_values_cost = self.cost.reshape(-1,1) + self.dic_agent_conf["GAMMA"]*soft_state_values_cost.reshape(-1,1)
    
            soft_q_values = self.action_att_predict(self.states, self.critic_local).gather(1, self.actions.reshape(-1,1))
            soft_q_values2 = self.action_att_predict(self.states, self.critic_local2).gather(1, self.actions.reshape(-1,1))
    
            critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values).to(device)
            critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values).to(device)
    
            critic_loss = critic_square_error.mean()
            critic2_loss = critic2_square_error.mean()
            # print("Critic Loss: ", critic_loss)

            critic_loss.backward()
            critic2_loss.backward()
            
            clip_grad_norm_(self.critic_local.parameters(), 0.5)
            clip_grad_norm_(self.critic_local2.parameters(), 0.5)

            self.critic_optimiser.step()    
            self.critic_optimiser2.step()
            
            if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain:
                soft_q_values_cost = self.action_att_predict(self.states, self.critic_local_cost).gather(1, self.actions.reshape(-1,1))
                soft_q_values_cost2 = self.action_att_predict(self.states, self.critic_local_cost2).gather(1, self.actions.reshape(-1,1))
                cse_cost = torch.nn.MSELoss(reduction="none")(soft_q_values_cost, next_q_values_cost).to(device)
                cse_cost2 = torch.nn.MSELoss(reduction="none")(soft_q_values_cost2, next_q_values_cost).to(device)        
                critic_loss_cost = cse_cost.mean()
                critic_loss_cost2 = cse_cost2.mean()
                
                self.critic_optimiser_cost.zero_grad()
                self.critic_optimiser_cost2.zero_grad()  
                
                critic_loss_cost.backward()
                critic_loss_cost2.backward()
                self.critic_optimiser_cost.step()
                self.critic_optimiser_cost2.step()
    
            actor_loss, log_action_probabilities = self.actor_loss(self.states)
            # print("Actor Loss: ", actor_loss)

            actor_loss.backward()
            clip_grad_norm_(self.actor_local.parameters(), 0.5)
            self.actor_optimiser.step()   

            self.soft_update_target_networks()
            
            alpha_loss = self.temperature_loss(log_action_probabilities)
            alpha_loss.backward()
            self.alpha_optimiser.step()
            # self.alpha = self.log_alpha.exp()
    
            if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain and k%self.lam_UR==0:
                
                lam_loss = self.lambda_loss()
                self.lam_optimiser.zero_grad()
                lam_loss.backward()
                self.lam_optimiser.step() 
    
            
        
    def actor_loss(self, states_tensor):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        with torch.no_grad():
            q_values_local = self.action_att_predict(states_tensor, self.critic_local)
            q_values_local2 = self.action_att_predict(states_tensor, self.critic_local2)
        inside_term = self.log_alpha.exp() * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        
        if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain:
            q_values_cost = self.action_att_predict(states_tensor, self.critic_local_cost)
            q_values_cost2 = self.action_att_predict(states_tensor, self.critic_local_cost2)
            penalty = self.lam * torch.min(q_values_cost, q_values_cost2).to(device)
            policy_loss = (action_probabilities * (inside_term+penalty)).sum(dim=1).mean()
            # policy_loss = (action_probabilities * (inside_term-penalty)).sum(dim=1).mean()
        else:
            policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
            
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        # print("Alpha Loss: ", alpha_loss)
        return alpha_loss   

    def lambda_loss(self):
        q_cost = self.action_att_predict(self.states, self.critic_local_cost).gather(1, self.actions.reshape(-1,1))
        q_cost2 = self.action_att_predict(self.states, self.critic_local_cost2).gather(1, self.actions.reshape(-1,1))
        violation = torch.min(q_cost, q_cost2).to(device)  - self.cost_lim
        
        self.log_lam = torch.nn.functional.softplus(self.lam).to(device)
        lambda_loss =  self.log_lam*violation.detach()
        lambda_loss = -lambda_loss.sum(dim=-1).mean()
        print("Lambda Loss: ", lambda_loss)

        return lambda_loss
    
    def get_action_info(self, states_tensor):
        
        action_probabilities = self.action_att_predict(states_tensor, self.actor_local)   
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def soft_update_target_networks(self, tau=0.01):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            self.soft_update(self.critic_target_cost, self.critic_local_cost, tau)
            self.soft_update(self.critic_target_cost2, self.critic_local_cost2, tau)
        
    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def build_memory(self):

        return []


    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        if self.cnt_round==0:
            file_path = self.dic_path["PATH_TO_PRETRAIN_MODEL"]
            
        self.critic_local.load_state_dict(torch.load(os.path.join(file_path, "%s_critic.h5" % file_name), map_location='cpu'))  
        self.critic_local2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic2.h5" % file_name), map_location='cpu'))  
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            self.critic_local_cost.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_cost.h5" % file_name), map_location='cpu'))  
            self.critic_local_cost2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_cost2.h5" % file_name), map_location='cpu'))  
        
        self.actor_local.load_state_dict(torch.load(os.path.join(file_path, "%s_actor.h5" % file_name), map_location='cpu'))  
        
        self.alpha = torch.load(os.path.join(file_path, "%s_alpha.pt" % file_name), map_location='cpu')        
        self.log_alpha = self.alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        
        self.lam = torch.load(os.path.join(file_path, "%s_lam.pt" % file_name), map_location='cpu')        
        self.lam_optimiser = torch.optim.Adam([self.lam], lr=self.lam_lr)
        
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        if self.cnt_round==0:
            file_path = self.dic_path["PATH_TO_PRETRAIN_MODEL"]

        self.critic_target.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target.h5" % file_name), map_location='cpu'))  
        self.critic_target2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target2.h5" % file_name), map_location='cpu'))  
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            self.critic_target_cost.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target_cost.h5" % file_name), map_location='cpu'))  
            self.critic_target_cost2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target_cost2.h5" % file_name), map_location='cpu'))  
        
        print("succeed in loading target model %s"%file_name) 

    def save_network(self, file_name):
        torch.save(self.actor_local.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_actor.h5" % file_name)) 
        torch.save(self.critic_local.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic.h5" % file_name))        
        torch.save(self.critic_local2.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic2.h5" % file_name))  
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            torch.save(self.critic_local_cost.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_cost.h5" % file_name))        
            torch.save(self.critic_local_cost2.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_cost2.h5" % file_name))  
        
        torch.save(self.alpha, os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_alpha.pt" % file_name))  
        torch.save(self.lam, os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_lam.pt" % file_name))  
        
    def save_network_bar(self, file_name):
        torch.save(self.critic_target.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target.h5" % file_name))        
        torch.save(self.critic_target2.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target2.h5" % file_name))        
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            torch.save(self.critic_target_cost.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target_cost.h5" % file_name))        
            torch.save(self.critic_target_cost2.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target_cost2.h5" % file_name))   
        