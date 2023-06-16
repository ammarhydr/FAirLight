import numpy as np 
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from agent import Agent
import random 
import torch
from torch.nn.utils import clip_grad_norm_
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# from torch import nn
# import torch.nn.functional as F

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

class SACAgentOne(Agent): 
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
        super(SACAgentOne, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,intersection_id)
                
        self.num_agents=dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_neighbors=min(dic_traffic_env_conf['TOP_K_ADJACENCY'],self.num_agents)

        self.num_actions = len(self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        # self.memory = self.build_memory()
        self.cnt_round = cnt_round
        
        # self.state_dim = 12 *2 + 1#* self.num_agents
        if dic_traffic_env_conf['NEIGHBOR']:
            self.state_dim = 12 * 10#* self.num_agents with neighbor
        else:
            self.state_dim = 12 + 4 #* self.num_agents
        self.target_entropy = 0.98 * -np.log(1 / self.num_actions)
        self.cost_lim = -10

        self.constrain = dic_traffic_env_conf['CONST_NUM']
        # self.alpha_lr=5e-06
        # self.cost_lim = -1#-5e-3
        # self.lam_lr=1e-7
        # self.actor_lr=0.0005
        
        # # single worked
        # self.alpha_lr = 5e-05
        # self.lam_lr = 1e-3
        # self.actor_lr = 1e-3

        self.alpha_lr = self.dic_agent_conf["ALPHA_LEARNING_RATE"]
        self.lam_lr = self.dic_agent_conf["LAM_LEARNING_RATE"]
        self.actor_lr = self.dic_agent_conf["ACTOR_LEARNING_RATE"]
        self.lam_UR = int(self.dic_agent_conf["LAM_UPDATE_RATE"])

        if cnt_round == 0: 
            # initialization
            self.critic_local = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
            self.critic_local2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
            self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
    
            self.critic_target = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
            self.critic_target2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)

            self.actor_local = Network(input_dimension=self.state_dim, output_dimension=self.num_actions, output_activation=torch.nn.Softmax(dim=1)).to(device)
            self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)
            self.c_actions=[1,2,3,4]
            

            if self.dic_traffic_env_conf["CONSTRAINT"]:
                self.critic_local_cost = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
                self.critic_local_cost2= Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
                self.critic_optimiser_cost = torch.optim.Adam(self.critic_local_cost.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
                self.critic_optimiser_cost2 = torch.optim.Adam(self.critic_local_cost2.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
        
                self.critic_target_cost = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
                self.critic_target_cost2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
    
            self.soft_update_target_networks(tau=1.)      

            # self.load_network("round_{0}_inter_{1}".format(499, self.intersection_id))
            # self.load_network_bar("round_{0}_inter_{1}".format(499, self.intersection_id))
            
            self.log_alpha = torch.tensor(np.log(1.), requires_grad=True)
            self.alpha = self.log_alpha
            self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

            self.lam = torch.tensor(1.0, requires_grad=True)
            self.lam_optimiser = torch.optim.Adam([self.lam], lr=self.lam_lr)
        else:
            # initialization
            self.critic_local = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
            self.critic_local2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
            self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
            self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
    
            self.critic_target = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
            self.critic_target2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
    
            self.actor_local = Network(input_dimension=self.state_dim, output_dimension=self.num_actions, output_activation=torch.nn.Softmax(dim=1)).to(device)
            self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)


            if self.dic_traffic_env_conf["CONSTRAINT"]:    
                self.critic_local_cost = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
                self.critic_local_cost2= Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
                self.critic_optimiser_cost = torch.optim.Adam(self.critic_local_cost.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
                self.critic_optimiser_cost2 = torch.optim.Adam(self.critic_local_cost2.parameters(), lr=self.dic_agent_conf["CRITIC_LEARNING_RATE"])
        
                self.critic_target_cost = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
                self.critic_target_cost2 = Network(input_dimension=self.state_dim, output_dimension=self.num_actions).to(device)
    
            self.soft_update_target_networks(tau=1.)
            self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
            self.load_network_bar("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round)
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

        # if self.dic_traffic_env_conf["CONSTRAINT"] and cnt_round > self.constrain:
        #     decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"], cnt_round-self.constrain)
        #     self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])  
            
            
            
        #     decayed_lr = self.dic_agent_conf["CRITIC_LEARNING_RATE"] * pow(self.dic_agent_conf["LEARNING_DECAY"], cnt_round)
        #     self.dic_agent_conf["CRITIC_LEARNING_RATE"] = max(decayed_lr, self.dic_agent_conf["MIN_LEARNING_RATE"])
            
        #     decayed_lr = self.dic_agent_conf["ACTOR_LEARNING_RATE"] * pow(self.dic_agent_conf["LEARNING_DECAY"], cnt_round)
        #     self.dic_agent_conf["ACTOR_LEARNING_RATE"] = max(decayed_lr, self.dic_agent_conf["MIN_LEARNING_RATE"])

        # print(self.alpha)

    def choose_action(self, count, state):
        '''
        choose the best action for current state
        -input: state:[[state inter1],[state inter1]]
        -output: act: [#agents,num_actions]
        '''


        feature=[]
        for j in range(self.num_agents):
            observation=[]
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "cur_phase":
                    observation.extend(self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                                    [state[j][feature_name][0]])
                else:
                    observation.extend(state[j][feature_name])
            feature.append(observation)

        
        state_input=np.array(feature)
        action_probabilities = self.get_action_probabilities(state_input)
        # discrete_action = [np.random.choice(range(self.num_actions), p=probs) for probs in action_probabilities]
        # discrete_action = [np.random.choice(range(self.num_actions), p=action_probabilities)]
        if np.random.rand() <= self.dic_agent_conf["EPSILON"]:
            discrete_action = [np.random.randint(0, self.num_actions)]
        else:
            discrete_action = [np.argmax(action_probabilities,axis=-1)]

        self.c_actions.append(discrete_action[0])
        # if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain and len(set(self.c_actions[-4:]))==1:
        # if self.dic_traffic_env_conf["CONSTRAINT"] and len(set(self.c_actions[-4:]))==1:
        # # if len(set(self.c_actions[-4:]))==1 and self.cnt_round>self.constrain:
        #     # # print('Constrain violated action: ',discrete_action)
        #     self.c_actions.pop()
        #     action_probabilities[discrete_action]=0
        #     discrete_action = [np.argmax(action_probabilities)]
        #     # if self.c_actions[-1]==0:
        #     #     discrete_action=[2]
        #     # elif self.c_actions[-1]==2:
        #     #     discrete_action=[0]
        #     # elif self.c_actions[-1]==1:
        #     #     discrete_action=[3]            
        #     # elif self.c_actions[-1]==3:
        #     #     discrete_action=[1]
        #     self.c_actions.append(discrete_action[0])
        #     # # print('New action: ',discrete_action)
        return discrete_action

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).cpu().detach().numpy()
    

    def reform_state(self, state):

        dic_state_feature_arrays = {} # {feature1: [inter1, inter2,..], feature2: [inter1, inter 2...]}
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []


        # for s in states:
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name].append(np.array(state[feature_name]))

        state_input = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in
                       self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]      
        
        return state_input

    def prepare_Xs_Y(self, memory, dic_exp_conf):
        
        self.memory=memory

    def train_network(self, dic_exp_conf):

        ind_end = len(self.memory)
        print("memory size: {0}".format(ind_end))

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = self.memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))


        epochs = self.dic_agent_conf["EPOCHS"]
        for k in range(epochs):   
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
                _action.append([])
                _reward.append([])
                _constraint.append([])
                for j in range(self.num_agents):
                    state, action, next_state, reward, _, constraint = sample_slice[i][j]
                    _state[i].append(np.concatenate(self.reform_state(state), axis=1))
                    _next_state[i].append(np.concatenate(self.reform_state(next_state), axis=1))
                    _action[i].append(action)
                    _reward[i].append(reward)
                    _constraint[i].append(constraint)
    
            self.states_tensor = torch.tensor(np.array(_state, dtype=np.float32).squeeze(2)).reshape(self.num_agents, len(_state), -1).to(device)
            self.actions_tensor = torch.tensor(_action).reshape(self.num_agents, len(_state)).to(device)
            self.rewards_tensor = torch.tensor(_reward).float().reshape(self.num_agents, len(_state)).squeeze(1).to(device)
            self.cost_tensor = torch.tensor(_constraint).float().reshape(self.num_agents, len(_state)).squeeze(1).to(device)
            self.next_states_tensor = torch.tensor(np.array(_next_state, dtype=np.float32).squeeze(2)).reshape(self.num_agents, len(_state), -1).to(device)
    
            for i in range(self.num_agents):
    
                with torch.no_grad():
                    action_probabilities, log_action_probabilities = self.get_action_info(self.next_states_tensor[i])
                    
                    next_q_values_target = self.critic_target.forward(self.next_states_tensor[i])
                    next_q_values_target2 = self.critic_target2.forward(self.next_states_tensor[i])
                    soft_state_values = (action_probabilities * (torch.min(next_q_values_target, next_q_values_target2) - self.log_alpha.exp() * log_action_probabilities)).sum(dim=1)
                    next_q_values = self.rewards_tensor[i] + self.dic_agent_conf["GAMMA"]*soft_state_values    
                    
                    if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain:
                        next_q_values_target_cost = self.critic_target_cost.forward(self.next_states_tensor[i])
                        next_q_values_target_cost2 = self.critic_target_cost2.forward(self.next_states_tensor[i])
                        soft_state_values_cost = (action_probabilities * (torch.min(next_q_values_target_cost, next_q_values_target_cost2) - self.log_alpha.exp() * log_action_probabilities)).sum(dim=1)
                        next_q_values_cost = self.cost_tensor[i] + self.dic_agent_conf["GAMMA"]*soft_state_values_cost
        
                soft_q_values = self.critic_local(self.states_tensor[i]).gather(1, self.actions_tensor[i].reshape(-1,1)).squeeze(-1)
                soft_q_values2 = self.critic_local2(self.states_tensor[i]).gather(1, self.actions_tensor[i].reshape(-1,1)).squeeze(-1)   
    
                critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
                critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        
                critic_loss = critic_square_error.mean()
                critic2_loss = critic2_square_error.mean()
                critic_loss.backward()
                critic2_loss.backward()

                clip_grad_norm_(self.critic_local.parameters(), 2)
                clip_grad_norm_(self.critic_local2.parameters(), 2)
    
                self.critic_optimiser.step()
                self.critic_optimiser.zero_grad()
    
                self.critic_optimiser2.step()
                self.critic_optimiser2.zero_grad()  

                actor_loss, log_action_probabilities = self.actor_loss(self.states_tensor[i])
                actor_loss.backward()
                clip_grad_norm_(self.actor_local.parameters(), 2)
                self.actor_optimiser.step()   
                self.actor_optimiser.zero_grad()
                
                if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain:
                    soft_q_values_cost = self.critic_local_cost(self.states_tensor[i]).gather(1, self.actions_tensor[i].reshape(-1,1)).squeeze(-1)
                    soft_q_values_cost2 = self.critic_local_cost2(self.states_tensor[i]).gather(1, self.actions_tensor[i].reshape(-1,1)).squeeze(-1)
                    cse_cost = torch.nn.MSELoss(reduction="none")(soft_q_values_cost, next_q_values_cost)
                    cse_cost2 = torch.nn.MSELoss(reduction="none")(soft_q_values_cost2, next_q_values_cost)        
                    critic_loss_cost = cse_cost.mean()
                    critic_loss_cost2 = cse_cost2.mean()
                    
                    critic_loss_cost.backward()
                    critic_loss_cost2.backward()

                    clip_grad_norm_(self.critic_local_cost.parameters(), 2)
                    clip_grad_norm_(self.critic_local_cost2.parameters(), 2)

                    self.critic_optimiser_cost.step()
                    self.critic_optimiser_cost2.step()
                    
                    if k%self.lam_UR==0:
                        lam_loss = self.lambda_loss(i)
                        lam_loss.backward()
                        self.lam_optimiser.step() 
                        
            self.soft_update_target_networks()

            alpha_loss = self.temperature_loss(log_action_probabilities)
            alpha_loss.backward()
            self.alpha_optimiser.step()
            # self.alpha = self.log_alpha.exp()
            
        print("Actor Loss: ", actor_loss)
        print("Critic Loss: ", critic_loss)      
        
    def actor_loss(self, states_tensor):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = (self.log_alpha.exp() * log_action_probabilities - torch.min(q_values_local, q_values_local2))
        
        if self.dic_traffic_env_conf["CONSTRAINT"] and self.cnt_round>self.constrain:
            q_values_cost = self.critic_local_cost(states_tensor)
            q_values_cost2 = self.critic_local_cost2(states_tensor)
            penalty = self.lam * torch.min(q_values_cost, q_values_cost2)
            policy_loss = (action_probabilities * (inside_term+penalty)).sum(dim=1).mean()
        else:
            policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        # print("Actor Loss: ", policy_loss)
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        # print("Alpha Loss: ", alpha_loss)
        return alpha_loss   

    def lambda_loss(self, i):
        q_cost = self.critic_local_cost(self.states_tensor[i]).gather(1, self.actions_tensor[i].reshape(-1,1)).squeeze(-1)
        q_cost2 = self.critic_local_cost2(self.states_tensor[i]).gather(1, self.actions_tensor[i].reshape(-1,1)).squeeze(-1)
        violation = torch.min(q_cost, q_cost2)  - self.cost_lim
        
        self.log_lam = torch.nn.functional.softplus(self.lam)
        lambda_loss =  self.log_lam*violation.detach()
        lambda_loss = -lambda_loss.mean(dim=-1)
        print("Lambda Loss: ", lambda_loss)

        return lambda_loss
    
    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def soft_update_target_networks(self, tau=0.03):
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
        
        open_file = open(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_c_actions" % file_name), "rb")
        self.c_actions = pickle.load(open_file)
        open_file.close()
        
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
        
        open_file = open(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_c_actions" % file_name), "wb")
        pickle.dump(self.c_actions, open_file)
        open_file.close()
        
        
    def save_network_bar(self, file_name):
        torch.save(self.critic_target.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target.h5" % file_name))        
        torch.save(self.critic_target2.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target2.h5" % file_name))        
        if self.dic_traffic_env_conf["CONSTRAINT"]:
            torch.save(self.critic_target_cost.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target_cost.h5" % file_name))        
            torch.save(self.critic_target_cost2.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_target_cost2.h5" % file_name))   
