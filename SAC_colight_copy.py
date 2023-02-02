import numpy as np 
import os 
from agent import Agent
import random 
import torch
from torch import nn
# import torch.utils.data as data
import torch.nn.functional as F


from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
import torch_scatter
from collections import OrderedDict


# import ray
# from ray import tune
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ColightNet(torch.nn.Module):
    def __init__(self, obs_space, action_space, model_config, name):
        super(ColightNet, self).__init__()
        # TorchModelV2.__init__(self, obs_space, action_space,action_space, model_config, name)
        # nn.Module.__init__(self)
        # neighbor have to be min(num_agents, num_neighbors) if neighbors should be adjusted for test purposes
        self.name=name
        self.num_agents=model_config['NUM_INTERSECTIONS']
        self.num_neighbors=min(model_config['TOP_K_ADJACENCY'],self.num_agents)
        
        self.num_actions = len(model_config["PHASE"][model_config['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(model_config["LANE_NUM"].values())))


        self.state_dim = 12 + 8 #* self.num_agents


        # dimension oriented at official CoLight implementation
        # self.dimension = 32
        # self.cnn_layer = [[32, 32]]
        # self.cnn_heads = [1]
        # self.mlp_layer = [128,128]

        # self.mham_layers = nn.ModuleList().to(device)
        # MLP, feature, dimension
        # self.mlp = MLP(self.state_dim, self.mlp_layer).to(device)
        # self.mlp = Network(self.state_dim, self.mlp_layer[1]).to(device)
        
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
        

       #  # num of intersections, neighbor representation
        # neighbor = torch.Tensor(self.num_agents, self.num_neighbors, self.num_agents).to(device)

        # # for CNN_layer_index, CNN_layer_size in enumerate(self.cnn_layer):
        # #      mham = MHAM(self.num_agents, neighbor, self.num_actions, self.cnn_layer, self.num_neighbors, self.state_dim,
        # #                         self.dimension, CNN_layer_size[0], self.cnn_heads[CNN_layer_index],
        # #                         CNN_layer_size[1]).to(device)
        # #      self.mham_layers.append(mham)

        # self.mham = MHAM(self.num_agents, neighbor,self.num_actions, self.cnn_layer, self.num_neighbors, self.state_dim,
        #                   self.dimension, self.cnn_layer[0][0], self.cnn_heads[0], self.cnn_layer[0][1]).to(device)
        # self.out_hidden_layer = nn.Linear(self.cnn_layer[-1][1], self.num_actions).to(device)


    #def forward(self, nei, nei_actions, agent, actions):
    def forward(self, x, edge_index):
        # agent = torch.tensor(agent, dtype=torch.float32).to(device)
        # adj_t = torch.tensor(state[1], dtype=torch.float32).to(device)
        # edge_index = adj_t.nonzero().t().contiguous()

        # dp = Data(x=agent, edge_index=edge_index)
        h = self.embedding_MLP.forward(x)
        for mdl in self.modulelist:
            h = mdl.forward(h, edge_index)
        x = self.output_layer(h)

        # batch_size = agent.shape[0]
        # att_record = []
        #agent = torch.from_numpy(agent).float()
        # x = self.mlp(agent)
        # att_record_all_layers = []
        # for i, mham in enumerate(self.mham_layers):
        # x = self.mham(x, edge_index)
        # att_record_all_layers = att_record
        # if len(self.cnn_layer) > 1:
        #     att_record_all_layers = torch.cat(att_record_all_layers, dim=1)
        # else:
        #     att_record_all_layers = att_record_all_layers[0]
        # att_record = torch.reshape(att_record_all_layers, (batch_size, len(self.cnn_layer), self.num_agents, self.cnn_heads[-1], self.num_neighbors))
        # x = self.out_hidden_layer(x)
        # x = x[:,0,:]
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

# LambdaLayer for mimic Keras.Lambda layer
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# see CoLight 4.1 (https://dl.acm.org/doi/10.1145/3357384.3357902)
class MLP(nn.Module):
    def __init__(self, input_shape, layer):
        super(MLP, self).__init__()
        layers = []
        for layer_index, layer_size in enumerate(layer):
            if layer_index == 0:
                layers.append(nn.Linear(input_shape, layer_size).to(device))
                layers.append(nn.ReLU().to(device))
            else:
                layers.append(nn.Linear(layer[layer_index - 1], layer_size).to(device))
                layers.append(nn.ReLU().to(device))

        self.seq = nn.Sequential(*layers).to(device)

    def forward(self, ob):
        x = self.seq(ob)
        return x
    
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



# see CoLight 4.2 (https://dl.acm.org/doi/10.1145/3357384.3357902)
class MHAM(nn.Module):

    def __init__(self, num_agents, neighbor, action_space, cnn_layer, num_neighbors, input_shape=24, dimension=128, dv=16, nv=8, dout=128):
        super(MHAM, self).__init__()
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.dimension = dimension
        self.dv = dv
        self.nv = nv
        #self.neighbor = neighbor
        self.feature_length = input_shape
        self.dout = dout
        self.action_space = action_space

        # [agent,1,dim]->[agent,1,dv*nv], since representation of specific agent=1
        self.agent_head_hidden_layer = nn.Linear(self.dimension, self.dv*self.nv)
        self.agent_head_lambda_layer = LambdaLayer((lambda x: x.permute(0,1,4,2,3)))

       # self.neighbor_repr_3D = RepeatVector3D(num_agents)
        # [agent,neighbor,agent]x[agent,agent,dim]->[agent,neighbor,dim]
        #self.neighbor_repr_lambda_layer = LambdaLayer((lambda x: torch.einsum('ana, aad -> and', x[0], x[1])))
        self.neighbor_repr_lambda_layer = LambdaLayer((lambda x: torch.matmul(x[0], x[1])))

        # representation for all neighbors
        self.neighbor_repr_head_hidden_layer = nn.Linear(in_features=dv*nv, out_features=dv*nv)
        self.neighbor_repr_head_lambda_layer = LambdaLayer((lambda x: x.permute(0,1,4,2,3)))

        # [batch,agent,nv,1,dv]x[batch,agent,nv,neighbor,dv]->[batch,agent,nv,1,neighbor]
        self.attention_layer = LambdaLayer((lambda x: F.softmax(torch.einsum('bancd, baned -> bance', x[0], x[1]))))

        # self embedding
        self.neighbor_hidden_repr_head_hidden_layer = nn.Linear(dv*nv, dv*nv)
        self.neighbor_hidden_repr_head_lambda_layer = LambdaLayer((lambda x: x.permute(0,1,4,2,3)))
        # mean values, preserving tensor shape
        self.out_lambda_layer = LambdaLayer((lambda x: torch.mean(torch.matmul(x[0], x[1]), 2)))
        self.out_hidden_layer = nn.Linear(dv, dout)

    def forward(self, agent, nei):
        batch_size = agent.size()[0]
        agent_repr = torch.reshape(agent, (batch_size, self.num_agents, 1, self.dimension))
        neighbor_repr = torch.reshape(agent, (batch_size, 1, self.num_agents, self.dimension))
        neighbor_repr = torch.tile(neighbor_repr, (1, self.num_agents,1,1))
        # nei = torch.FloatTensor(nei)
        #neighbor_repr = nei #self.neighbor_repr_lambda_layer([nei, neighbor_repr])
        neighbor_repr = self.neighbor_repr_lambda_layer([nei, neighbor_repr])

        agent_repr_head = self.agent_head_hidden_layer(agent_repr)
        agent_repr_head = F.relu(agent_repr_head)
        agent_repr_head = torch.reshape(agent_repr_head, (batch_size, self.num_agents, 1, self.dv, self.nv))

        agent_repr_head = self.agent_head_lambda_layer(agent_repr_head)
        neighbor_repr_head = self.neighbor_repr_head_hidden_layer(neighbor_repr)
        neighbor_repr_head = F.relu(neighbor_repr_head)
        # second num_agents could be replaced with num_neighbors if min(num_agents, num_neighbors)
        neighbor_repr_head = torch.reshape(neighbor_repr_head, (batch_size, self.num_agents, self.num_neighbors, self.dv, self.nv))
        neighbor_repr_head = self.neighbor_repr_head_lambda_layer(neighbor_repr_head)

       # agent_repr_head = agent_repr_head.reshape(-1, self.nv, 1, self.dv)
       # neighbor_repr_head = neighbor_repr_head.reshape(self.num_agents, self.nv, -1, self.dv)

        att = self.attention_layer([agent_repr_head, neighbor_repr_head])
        # second num_agents could be replaced with num_neighbors if min(num_agents, num_neighbors)
        # att_record = torch.reshape(att, (batch_size, self.num_agents, self.nv, self.num_neighbors))

        neighbor_hidden_repr_head = self.neighbor_hidden_repr_head_hidden_layer(neighbor_repr)
        neighbor_hidden_repr_head = F.relu(neighbor_hidden_repr_head)
        neighbor_hidden_repr_head = torch.reshape(neighbor_hidden_repr_head, (batch_size, self.num_agents, self.num_neighbors, self.dv, self.nv))
        neighbor_hidden_repr_head = self.neighbor_hidden_repr_head_lambda_layer(neighbor_hidden_repr_head)
        out = self.out_lambda_layer([att, neighbor_hidden_repr_head])
        out = torch.reshape(out, (batch_size,self.num_agents, self.dv))
        out = self.out_hidden_layer(out)
        out = F.relu(out)
        return out#, att_record


# Repeat vector x times
class RepeatVector3D(nn.Module):

    def __init__(self, times):
        super(RepeatVector3D, self).__init__()
        self.times = times

    def forward(self, x):
        x = torch.tile(torch.unsqueeze(x, 0), (1, self.times, 1, 1))
        return x
# SEED=6666
# random.seed(SEED)
# np.random.seed(SEED)
# torch.seed(SEED)


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
        # self.memory = self.build_memory()
        self.cnt_round = cnt_round
        
        # self.state_dim = 12 *2 + 1#* self.num_agents
        if dic_traffic_env_conf['NEIGHBOR']:
            self.state_dim = 12 * 10#* self.num_agents with neighbor
        else:
            self.state_dim = 12 + 8 #* self.num_agents
        self.target_entropy = 0.98 * -np.log(1 / self.num_actions)

        self.alpha_lr=3e-03
        self.cost_lim = 10#-5e-5
        self.lam_lr=3e-06

        if cnt_round == 0: 
            # initialization
            self.critic_local = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            self.critic_local2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
            self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
    
            self.critic_target = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            self.critic_target2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
            
            if self.dic_traffic_env_conf["CONSTRAINT"]:
                self.critic_local_cost = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
                self.critic_local_cost2= ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
                self.critic_optimiser_cost = torch.optim.Adam(self.critic_local_cost.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
                self.critic_optimiser_cost2 = torch.optim.Adam(self.critic_local_cost2.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
        
                self.critic_target_cost = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
                self.critic_target_cost2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Critic').to(device)
    
            self.soft_update_target_networks(tau=1.)
    
            self.actor_local = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Actor').to(device)
            self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
    
            self.log_alpha = torch.tensor(np.log(1.), requires_grad=True)
            self.alpha = self.log_alpha
            self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

            self.lam = torch.tensor(1.0, requires_grad=True)
            self.lam_optimiser = torch.optim.Adam([self.lam], lr=self.lam_lr)

        else:
            # initialization
            self.critic_local = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
            self.critic_local2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
            self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
    
            self.critic_target = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
            self.critic_target2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)

            if self.dic_traffic_env_conf["CONSTRAINT"]:    
                self.critic_local_cost = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
                self.critic_local_cost2= ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
                self.critic_optimiser_cost = torch.optim.Adam(self.critic_local_cost.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
                self.critic_optimiser_cost2 = torch.optim.Adam(self.critic_local_cost2.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
        
                self.critic_target_cost = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
                self.critic_target_cost2 = ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Colight').to(device)
    
            # self.soft_update_target_networks(tau=1.)
    
            self.actor_local =  ColightNet(self.state_dim, self.num_actions, dic_traffic_env_conf, 'Actor').to(device)
            self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"])
            
            try:
                
                self.load_network("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))
                # print('init q_bar load')
                self.load_network_bar("round_{0}_inter_{1}".format(cnt_round-1, self.intersection_id))        
                # if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                #     if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                #         self.load_network_bar("round_{0}_inter_{1}".format(
                #             max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                #             self.intersection_id))
                #     else:
                #         self.load_network_bar("round_{0}_inter_{1}".format(
                #             max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                #             self.intersection_id))
                # else:
                #     self.load_network_bar("round_{0}_inter_{1}".format(
                #         max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))
            except:
                print("fail to load network, current round: {0}".format(cnt_round))
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
        action_probabilities =self.action_att_predict([state], self.actor_local)
        discrete_action = [np.random.choice(range(self.num_actions), p=probs) for probs in action_probabilities.cpu().detach().numpy()]
        return discrete_action, 1

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()
    

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
            _state.append([])
            _next_state.append([])
            _action.append([])
            _reward.append([])
            _constraint.append([])
            for j in range(self.num_agents):
                state, action, next_state, reward, constraint, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _action[i].append(action)
                _reward[i].append(reward)
                _constraint[i].append(constraint)
            

        self.states = _state #= torch.tensor(np.array(_state, dtype=np.float32).squeeze(2).reshape((len(_state)*self.num_agents, -1)))
        self.actions = torch.tensor(_action).to(device) # torch.tensor(_action).reshape(-1,1)
        self.rewards = torch.tensor(_reward).to(device) #torch.tensor(_reward).float().reshape(-1,1).squeeze(1)
        self.cost = torch.tensor(_constraint) #torch.tensor(_constraint).float().reshape(-1,1).squeeze(1)
        self.next_states = _next_state #torch.tensor(np.array(_next_state, dtype=np.float32).squeeze(2).reshape((len(_state)*self.num_agents, -1)))

    def train_network(self, dic_exp_conf):
        
        # epochs = self.dic_agent_conf["EPOCHS"]
        # for epoch in range(epochs):

        # Train model
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(self.next_states)
            
            next_q_values_target = self.action_att_predict(self.next_states, self.critic_target)  
            next_q_values_target2 = self.action_att_predict(self.next_states, self.critic_target2)   

            soft_state_values = (action_probabilities * (torch.min(next_q_values_target, next_q_values_target2).to(device) - self.alpha * log_action_probabilities)).sum(dim=1)
            next_q_values = self.rewards.reshape(-1,1) + self.dic_agent_conf["GAMMA"]*soft_state_values.reshape(-1,1)    
            
            if self.dic_traffic_env_conf["CONSTRAINT"]:
                next_q_values_target_cost = self.action_att_predict(self.next_states, self.critic_target_cost) 
                next_q_values_target_cost2 = self.action_att_predict(self.next_states, self.critic_target_cost2)  
                soft_state_values_cost = (action_probabilities * (torch.min(next_q_values_target_cost, next_q_values_target_cost2).to(device) - self.log_alpha.exp() * log_action_probabilities)).sum(dim=1)
                next_q_values_cost = self.cost + self.dic_agent_conf["GAMMA"]*soft_state_values_cost

        soft_q_values = self.action_att_predict(self.states, self.critic_local).gather(1, self.actions.reshape(-1,1))
        soft_q_values2 = self.action_att_predict(self.states, self.critic_local2).gather(1, self.actions.reshape(-1,1))

        critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values).to(device)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values).to(device)

        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        critic_loss.backward()
        critic2_loss.backward()

        self.critic_optimiser.step()
        self.critic_optimiser.zero_grad()

        self.critic_optimiser2.step()
        self.critic_optimiser2.zero_grad()  
        
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            soft_q_values_cost = self.action_att_predict(self.states, self.critic_local_cost).gather(1, self.actions.reshape(-1,1))
            soft_q_values_cost2 = self.action_att_predict(self.states, self.critic_local_cost2).gather(1, self.actions.reshape(-1,1))
            cse_cost = torch.nn.MSELoss(reduction="none")(soft_q_values_cost, next_q_values_cost).to(device)
            cse_cost2 = torch.nn.MSELoss(reduction="none")(soft_q_values_cost2, next_q_values_cost).to(device)        
            critic_loss_cost = cse_cost.mean()
            critic_loss_cost2 = cse_cost2.mean()
            
            critic_loss_cost.backward()
            critic_loss_cost2.backward()
            self.critic_optimiser_cost.step()
            self.critic_optimiser_cost2.step()

        actor_loss, log_action_probabilities = self.actor_loss(self.states)
        actor_loss.backward()
        self.actor_optimiser.step()   
        self.actor_optimiser.zero_grad()
        
        alpha_loss = self.temperature_loss(log_action_probabilities)
        alpha_loss.backward()
        self.alpha_optimiser.step()

        if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round%(self.dic_agent_conf["UPDATE_Q_BAR_FREQ"])==0:
            
            lam_loss = self.lambda_loss()
            lam_loss.backward()
            self.lam_optimiser.step() 

        self.soft_update_target_networks()
          
        
    def actor_loss(self, states_tensor):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.action_att_predict(states_tensor, self.critic_local)
        q_values_local2 = self.action_att_predict(states_tensor, self.critic_local2)
        inside_term = (self.log_alpha.exp() * log_action_probabilities - torch.min(q_values_local, q_values_local2).to(device))
        
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            q_values_cost = self.action_att_predict(states_tensor, self.critic_local_cost)
            q_values_cost2 = self.action_att_predict(states_tensor, self.critic_local_cost2)
            penalty = self.lam * torch.min(q_values_cost, q_values_cost2).to(device)
            policy_loss = (action_probabilities * (inside_term+penalty)).sum(dim=1).mean()
        else:
            policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
            
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss   

    def lambda_loss(self):
        q_cost = self.action_att_predict(self.states, self.critic_local_cost).gather(1, self.actions.reshape(-1,1))
        q_cost2 = self.action_att_predict(self.states, self.critic_local_cost2).gather(1, self.actions.reshape(-1,1))
        violation = torch.min(q_cost, q_cost2).to(device)  - self.cost_lim
        
        self.log_lam = torch.nn.functional.softplus(self.lam).to(device)
        lambda_loss =  self.log_lam*violation.detach()
        lambda_loss = -lambda_loss.sum(dim=-1).mean()
        return lambda_loss
    
    def get_action_info(self, states_tensor):
        
        action_probabilities=self.action_att_predict(states_tensor, self.actor_local)   
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
            
        self.critic_local.load_state_dict(torch.load(os.path.join(file_path, "%s_critic.h5" % file_name)))  
        self.critic_local2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic2.h5" % file_name)))  
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            self.critic_local_cost.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_cost.h5" % file_name)))  
            self.critic_local_cost2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_cost2.h5" % file_name)))  
        
        self.actor_local.load_state_dict(torch.load(os.path.join(file_path, "%s_actor.h5" % file_name)))  
        
        self.alpha = torch.load(os.path.join(file_path, "%s_alpha.pt" % file_name))        
        self.log_alpha = self.alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        
        self.lam = torch.load(os.path.join(file_path, "%s_lam.pt" % file_name))        
        self.lam_optimiser = torch.optim.Adam([self.lam], lr=self.lam_lr)
        
        print("succeed in loading model %s"%file_name)

    def load_network_bar(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.critic_target.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target.h5" % file_name)))  
        self.critic_target2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target2.h5" % file_name)))  
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            self.critic_target_cost.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target_cost.h5" % file_name)))  
            self.critic_target_cost2.load_state_dict(torch.load(os.path.join(file_path, "%s_critic_target_cost2.h5" % file_name)))  
        
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



