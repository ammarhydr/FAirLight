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
class ColightNet(torch.nn.Module):
    def __init__(self, obs_space, action_space, model_config, name):
        super(ColightNet, self).__init__()

        self.name=name
        self.num_agents=model_config['NUM_INTERSECTIONS']
        self.num_neighbors=min(model_config['TOP_K_ADJACENCY'],self.num_agents)
        
        self.num_actions = len(model_config["PHASE"][model_config['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(model_config["LANE_NUM"].values())))


        self.state_dim = obs_space


        
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

# SEED=6666
# random.seed(SEED)
# np.random.seed(SEED)
# torch.seed(SEED)


class ColightTorch(Agent): 
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
        super(ColightTorch, self).__init__(
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

        # self.cost_lim = 10#-5e-5
        # self.lam_lr=3e-06

        if cnt_round == 0: 
            # initialization
            self.critic_local = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
    
            self.critic_target = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            

            self.soft_update_target_networks(tau=1.)


        else:
            # initialization
            self.critic_local = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
    
            self.critic_target = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)

            try:
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
                self.load_network_bar("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
                # self.load_network_bar("round_{0}_inter_{1}".format(max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))
                
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
            print("memory samples number:", sample_size)
    
            _state = []
            _next_state = []
            _action=[]
            _reward=[]
            _constraint=[]
            for i in range(len(sample_slice)):
                _state.append([])
                _next_state.append([])
                _constraint.append([])
                for j in range(self.num_agents):
                    state, action, next_state, reward, _, constraint = sample_slice[i][j]
                    _state[i].append(state)
                    _next_state[i].append(next_state)
                    _action.append(action)
                    _reward.append(reward)
                    _constraint[i].append(constraint)
                
    
            self.states = _state 
            self.actions = torch.tensor(_action).to(device) 
            self.rewards = torch.tensor(_reward).to(device) 
            self.cost = torch.tensor(_constraint) 
            self.next_states = _next_state 

            with torch.no_grad():
                out = self.action_att_predict(self.next_states, self.critic_target)
                target_q_values = self.rewards.reshape(-1,1) + self.dic_agent_conf["GAMMA"] * torch.max(out, dim=1)[0].reshape(-1,1)
                
            q_values = self.action_att_predict(self.states, self.critic_local).gather(1, self.actions.reshape(-1,1))

            loss = torch.nn.MSELoss(reduction="none")(q_values, target_q_values).mean()
            print("Loss: ", loss)
            self.critic_optimiser.zero_grad()
            loss.backward()
            clip_grad_norm_(self.critic_local.parameters(), 0.5)
            self.critic_optimiser.step()
        
            self.soft_update_target_networks()


    def soft_update_target_networks(self, tau=0.01):
        self.soft_update(self.critic_target, self.critic_local, tau)

        
    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def build_memory(self):

        return []


    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
            
        self.critic_local.load_state_dict(torch.load(os.path.join(file_path, "%s_critic.h5" % file_name)))  
        
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.critic_target.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target.h5" % file_name)))  

    def save_network(self, file_name):
        torch.save(self.critic_local.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic.h5" % file_name))        

        
    def save_network_bar(self, file_name):
        torch.save(self.critic_target.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target.h5" % file_name))        


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
    one_agent=SACAgentColight(
        dic_agent_conf=dic_agent_conf, 
        dic_traffic_env_conf=dic_traffic_env_conf, 
        dic_path=dic_path, 
        cnt_round=cnt_round,        
    )
    one_model=one_agent.build_network()
    one_agent.build_network_from_copy(one_model)



