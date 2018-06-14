from __future__ import print_function, division
import numpy as np, time, random, gym, sys, os, torch
from gym import spaces

sys.dont_write_bytecode = True
from load_graph import *
from status     import StatusLine
from renderer   import *

#==============================================================================
# ENVIRONMENT CLASS
#==============================================================================

class GraphEnv(object):
    def __init__(self, datafile, vision_batch, shifts, shift_radians, stride, elevator_radius,
            vision_init, workers, stutter_prob, noise_sigma, noise_theta, max_variations,
            curr_levels, curr_upd_int, curr_thresh, bump_penalty, headless):

        self.workers           = workers
        self.stutter_prob      = stutter_prob
        self.noise_sigma       = noise_sigma
        self.noise_theta       = noise_theta
        self.curriculum_levels = curr_levels
        self.curr_upd_int      = curr_upd_int
        self.curr_thresh       = curr_thresh
        self.max_variations    = max_variations
        self.elevator_radius   = elevator_radius
        self.headless          = headless
        self.bump_penalty      = bump_penalty

        # load graph from archive file
        graph = load_graph(datafile, vision_batch, shifts, shift_radians, stride, vision_init, max_variations)

        #======================================================================
        # make graph into matrices for fast lookups
        #======================================================================

        # normalization statistics
        self.emb_mu  = graph["embedding_mu" ]
        self.emb_sig = np.clip(graph["embedding_sig"], 1e-3, np.inf)

        # get information about the graph
        self.obs_size         = graph["obs_size"][0]
        self.num_locations    = len(graph["graph_nodes"].keys())
        self.num_shifts       = shifts+1
        self.max_obs_per_node = 5
        self.obs_variations   = (shifts+1)*self.max_obs_per_node

        # make lookup dict so we can get indices right
        self.location_to_id = {}
        for node_id in graph["graph_nodes"].keys(): self.location_to_id[str(node_id)] = len(self.location_to_id.keys())
        self.id_to_location = {v:k for k,v in self.location_to_id.items()}
        print("Made a lookup table for {} node ids to location numbers".format(len(self.id_to_location)))

        # pack observations into a single matrix
        self.observations = np.zeros((self.num_locations, self.obs_variations, 4, self.obs_size), dtype=np.float32)
        for l,node in enumerate(graph["graph_nodes"].values()):
            for j in range(self.max_obs_per_node):
                obs_idx = (len(node["observations"])//2+j)%len(node["observations"])
                for d,shifts in enumerate(graph["raw_nodes"][str(node["observations"][obs_idx])]["img"]):
                    for k,obs in enumerate(shifts):
                        self.observations[l,j*self.num_shifts+k,d] = (obs-self.emb_mu) / self.emb_sig
        self.observations = self.observations.reshape(-1, self.obs_size)
        print("Packed observations into a single matrix of shape {}".format(self.observations.shape))

        # pack observation gradients into a single matrix
        self.obs_gradient = np.zeros((self.num_locations, self.obs_variations, 4, self.obs_size), dtype=np.float32)
        for l,node in enumerate(graph["graph_nodes"].values()):
            for j in range(self.max_obs_per_node):
                obs_idx = (len(node["observations"])//2+j)%len(node["observations"])
                for d,shifts in enumerate(graph["raw_nodes"][str(node["observations"][obs_idx])]["grd"]):
                    for k,grd in enumerate(shifts):
                        self.obs_gradient[l,j*self.num_shifts+k,d] = grd / self.emb_sig
        self.obs_gradient = self.obs_gradient.reshape(-1, self.obs_size)
        print("Packed observation gradients into a single matrix of shape {}".format(self.obs_gradient.shape))

        # find all the nonlocal edges that we need to make into special actions
        cluster_destinations = set()
        nodes_with_nonlocal_edges = set()
        self.lookup_direction_from_cluster = {}
        for node_id, node in graph["graph_nodes"].items():
            for target_id, edge_type, edge_direction in node["edges"]:
                if edge_type == "nonlocal":
                    nodes_with_nonlocal_edges.add(str(node_id))
                    target_node = graph["graph_nodes"][str(target_id)]
                    cluster = target_node["cluster"]
                    cluster_destinations.add(cluster)
                    if cluster not in self.lookup_direction_from_cluster:
                        self.lookup_direction_from_cluster[cluster] = len(self.lookup_direction_from_cluster.keys())
        self.num_nonlocal_destinations = len(cluster_destinations)
        self.lookup_cluster_from_direction = {v:k for k,v in self.lookup_direction_from_cluster.items()}
        print("Action space augmented with {} nonlocal destinations".format(self.num_nonlocal_destinations))

        # connect all near-elevator nodes to the elevator edges so the agent doesn't need
        # to be exactly in front of an elevator in order to use it
        edges_added = 0
        for node_id, node in graph["graph_nodes"].items():
            for target_id in nodes_with_nonlocal_edges:
                d = graph["path_lengths"][str(node_id)][str(target_id)]
                if d > 0 and d <= self.elevator_radius:
                    for e in graph["graph_nodes"][str(target_id)]["edges"]:
                        if e[1] == "nonlocal" and not any([e[0] == ed[0] for ed in node["edges"]]):
                            node["edges"].append(e)
                            edges_added += 1
        print("Connected {} new elevator edges in radius".format(edges_added, self.elevator_radius))

        # pack edges into a single matrix
        num_edges  = 0
        self.edges = np.full((self.num_locations, 4 + self.num_nonlocal_destinations), -1, np.int32)
        for l,node in enumerate(graph["graph_nodes"].values()):
            for target_id, edge_type, edge_direction in node["edges"]:
                if   edge_type == "natural" or edge_type == "local":
                    direction = edge_direction
                elif edge_type == "nonlocal":
                    cluster = graph["graph_nodes"][str(target_id)]["cluster"]
                    direction = 4 + self.lookup_direction_from_cluster[cluster]
                if self.edges[l,direction] >= 0: print("WARNING: edge direction reuse ({},{})! This may indicate a problem with the graph.".format(l,edge_direction))
                self.edges[l,direction] = self.location_to_id[str(target_id)]
                num_edges += 1
        self.edges = self.edges.flatten()
        print("Packed {} edges into a single matrix".format(num_edges))

        # pack new-backward-after-traversing-edge into a single matrix
        self.new_backward = np.full((self.num_locations, 4 + self.num_nonlocal_destinations), 0, np.int32)
        for l,node in enumerate(graph["graph_nodes"].values()):
            for target_id, edge_type, edge_direction in node["edges"]:
                if edge_type == "natural" or edge_type == "local":
                    # get the resulting new orientation
                    new_edges = graph["graph_nodes"][str(target_id)]["edges"]
                    edges_back = filter(lambda e:str(e[0])==str(node["id"]), new_edges)
                    if len(edges_back) > 0: back_id, back_type, back_direction = edges_back[0]
                    else: back_direction = 0; print("Warning, no edge back from {} to {}".format(target_id, node["id"]))
                    self.new_backward[l,edge_direction] = back_direction
        self.new_backward = self.new_backward.flatten()
        print("Filled in the 'new backward' direction resulting from travel along each edge")

        # pack optimal lengths into a single matrix
        self.optimal_path_length = np.zeros((self.num_locations, self.num_locations), np.float32)
        for start in graph["graph_nodes"].keys():
            for end in graph["graph_nodes"].keys():
                l = graph["path_lengths"][start][end]
                sid = self.location_to_id[str(start)]
                eid = self.location_to_id[str(end)]
                self.optimal_path_length[sid,eid] = l
        self.optimal_path_length = self.optimal_path_length.flatten()
        self.max_path_length = np.max(self.optimal_path_length)
        print("Packed the optimal path lengths into a single matrix")

        print("Mean: ", np.mean(self.optimal_path_length))
        print("Max:  ", np.max (self.optimal_path_length))
        print("Std:  ", np.std (self.optimal_path_length))

        # construct a curriculum of goals of increasing difficulty
        min_path_length          = 5
        time_limit_factor        = 5
        self.curriculum_level    = 0
        self.episodes_completed  = 0
        curriculum_cachefile = ".cache/curriculum_{}.".format(datafile.replace("/","_")) + ".".join([str(v) for v in [min_path_length, self.curriculum_levels]]) + ".pytorch"
        if os.path.exists(curriculum_cachefile):
            curr_cached = torch.load(curriculum_cachefile)
            self.curriculum_goals = curr_cached["curriculum_goals"]
            self.curriculum_limit = curr_cached["curriculum_limit"]
            self.goals_available  = curr_cached["goals_available" ]
        else:
            with StatusLine("Building curriculum levels"):
                self.curriculum_goals = np.zeros((self.curriculum_levels, self.num_locations, self.num_locations), np.int32)
                self.curriculum_limit = np.zeros((self.curriculum_levels,), np.float32)
                self.goals_available  = np.zeros((self.curriculum_levels, self.num_locations), np.int32)
                for l in range(self.curriculum_levels):
                    curriculum = (l+1)/float(self.curriculum_levels)
                    gap = self.max_path_length - min_path_length
                    max_dist = int(curriculum * gap) + min_path_length
                    self.curriculum_limit[l] = time_limit_factor * max_dist
                    for i in range(self.num_locations):
                        k = 0
                        for j in range(self.num_locations):
                            L_opt = self.optimal_path_length[i*self.num_locations + j]
                            if L_opt > 0 and L_opt <= max_dist:
                                self.curriculum_goals[l,i,k] = j
                                k += 1
                        self.goals_available[l,i] = k
                        for h in range(k,self.num_locations):
                            self.curriculum_goals[l,i,h] = self.curriculum_goals[l,i,h%k]
                self.curriculum_goals = self.curriculum_goals.reshape(self.curriculum_levels, -1)
                torch.save(dict(curriculum_goals=self.curriculum_goals, curriculum_limit=self.curriculum_limit, goals_available=self.goals_available), curriculum_cachefile)
        print("Built a curriculum of goals of increasing difficulty of size {}".format(self.curriculum_levels))

        # pre-generate a bunch of iid gaussian noise to save time
        self._iid_index = 0
        self._iid_noise = np.random.normal(0,1,(3141,self.obs_size)).astype(np.float32)

        self.action_space      = spaces.Discrete(3 + self.num_nonlocal_destinations)
        self.observation_space = spaces.Box(-np.inf, np.inf, graph["obs_size"])

        if not self.headless: self.graph = graph

        #======================================================================
        # initialize worker states
        #======================================================================

        self.workers_done     = np.full((self.workers,), 1, np.bool)
        self.workers_timestep = np.full((self.workers,), 0, np.float32)
        self.workers_motions  = np.full((self.workers,), 0, np.float32)
        self.workers_backward = np.full((self.workers,), 0, np.int32)
        self.workers_g_back   = np.full((self.workers,), 0, np.int32)
        self.workers_location = np.full((self.workers,), 0, np.int32)
        self.workers_start    = np.full((self.workers,), 0, np.int32)
        self.workers_goal     = np.full((self.workers,), 0, np.int32)
        self.workers_goal_obs = np.full((self.workers,self.obs_size), 0, np.float32)
        self.workers_obs      = np.full((self.workers,self.obs_size), 0, np.float32)
        self.workers_prev_act = np.full((self.workers,self.action_space.n), 0, np.float32)
        self.workers_ep_limit = np.full((self.workers,), self.curriculum_limit[0], np.float32)
        self.workers_eplength = np.full((self.workers,), self.curriculum_limit[0], np.float32)
        self.workers_epreward = np.full((self.workers,), 0, np.float32)
        self.workers_epsuclen = np.full((self.workers,), 0, np.float32)
        self.workers_reward   = np.full((self.workers,), 0, np.float32)
        self.workers_opfrac   = np.full((self.workers,), 0, np.float32)
        self.workers_oplength = np.full((self.workers,), 0, np.float32)
        self.workers_noise    = np.full((self.workers,), 0, np.float32)
        self.workers_success  = np.full((self.workers,), 0, np.float32)
        self.workers_epsucc   = np.full((self.workers,), 0, np.float32)

        self.workers_backward[...] = np.random.randint(0,4,(self.workers,))
        self.workers_location[...] = np.random.randint(0,self.num_locations,(self.workers,))

        # print information about how much memory each field is using
        print("----------------------")
        print("Memory usage:")
        for k,v in self.__dict__.items():
            if type(v) is np.ndarray:
                print("{:20} : {:.03f} megabytes".format(k,v.nbytes/1e6))
        print("----------------------")

    #--------------------------------------------------------------------------

    def update_curriculum_level(self):
        if self.episodes_completed < self.curr_upd_int: return
        self.episodes_completed = 0
        if np.mean(self.workers_epsucc) > self.curr_thresh:
            self.curriculum_level = min(self.curriculum_level+1, self.curriculum_levels-1)

    #--------------------------------------------------------------------------

    def reset(self, goal_loc=None, start_loc=None):
        if self.workers_done[0]: self.worker_0_path = []
        else: self.worker_0_path.append(self.workers_location[0])

        if goal_loc is not None or start_loc is not None: self.workers_done[...] = 1

        wd = self.workers_done
        num_done = wd.sum()
        self.episodes_completed += num_done
        if num_done > 0:
            cl = self.curriculum_level
            ws = np.logical_and(wd,self.workers_success>0) # successful workers

            self.workers_noise   [wd] = 0
            self.workers_epsuclen[wd] = self.workers_success[wd]*self.optimal_path_length[self.workers_start[wd]*self.num_locations + self.workers_goal[wd]]

            if goal_loc is not None:
                self.workers_goal    [wd] = goal_loc
                self.workers_g_back  [wd] = 0
            else:
                self.workers_goal    [wd] = self.curriculum_goals[cl][self.workers_location[wd]*self.num_locations + np.random.randint(0,self.num_locations,(num_done,))]
                self.workers_g_back  [wd] = np.random.randint(0,4,(num_done,))

            if start_loc is not None:
                self.workers_location[wd] = start_loc
                self.workers_backward[wd] = 0
                self.workers_start   [wd] = start_loc
            else:
                self.workers_location[wd] = np.random.randint(0,self.num_locations,(num_done,))
                self.workers_backward[wd] = np.random.randint(0,4,(num_done,))
                self.workers_start   [wd] = self.workers_location[wd]

            self.workers_goal_obs[wd] = self._obs_at(self.workers_goal    [wd], self.workers_g_back  [wd], wd)
            self.workers_obs     [wd] = self._obs_at(self.workers_location[wd], self.workers_backward[wd], wd)
            self.workers_ep_limit[wd] = self.curriculum_limit[cl]
            self.workers_eplength[wd] = self.workers_motions [wd]
            self.workers_opfrac  [ws] = self.workers_eplength[ws] / np.clip(self.workers_oplength[ws],1,np.inf)
            self.workers_epreward[wd] = self.workers_reward  [wd]
            self.workers_epsucc  [wd] = self.workers_success [wd]
            self.workers_oplength[wd] = self.optimal_path_length[self.workers_start[wd]*self.num_locations + self.workers_goal[wd]]
            self.workers_prev_act[wd] = 0
            self.workers_timestep[wd] = 0
            self.workers_motions [wd] = 0
            self.workers_reward  [wd] = 0
            self.workers_done    [wd] = 0
            self.workers_success [wd] = 0

        return (self.workers_obs.copy(),
                self.workers_goal_obs.copy(),
                self.workers_prev_act.copy(),
                self.workers_location.copy(),
                self.workers_goal.copy())

    #--------------------------------------------------------------------------

    def step(self, action):
        self.workers_timestep += 1

        moved  = np.full((self.workers,), False, np.bool)
        bumped = np.full((self.workers,), False, np.bool)

        turn_left  = action==0
        go_forward = np.logical_and(action == 1, np.random.random(action.shape) > self.stutter_prob)
        turn_right = action==2

        if turn_left .sum() > 0: self.workers_backward[turn_left ] = (self.workers_backward[turn_left ]-1)%4
        if turn_right.sum() > 0: self.workers_backward[turn_right] = (self.workers_backward[turn_right]+1)%4

        moved[turn_left ] = True
        moved[turn_right] = True

        En = 4 + self.num_nonlocal_destinations

        # forward local edges
        if go_forward.sum() > 0:
            edgeids = self.workers_location[go_forward]*En + (self.workers_backward[go_forward]+2)%4
            targets = self.edges[edgeids]
            valid   = targets>=0 # edges that don't exist have a target of -1
            if valid.sum() > 0:
                new_backward = self.new_backward[edgeids[valid]]
                all_targets = self.edges[self.workers_location*En + (self.workers_backward+2)%4]
                # mask indices must be flat or a copy is made and the state doesn't mutate
                w_valid = np.logical_and(go_forward, all_targets>=0)
                self.workers_location[w_valid] = all_targets[w_valid]
                self.workers_backward[w_valid] = new_backward
                self.workers_motions [w_valid] = self.workers_motions[w_valid] + 1
                moved[w_valid]     = True
                bumped[go_forward] = np.logical_not(w_valid[go_forward])

        # elevator edges
        for nonlocal_action in range(self.num_nonlocal_destinations):
            # adding 3 because the first 3 actions are left/forward/right
            action_selected = action==(3+nonlocal_action)
            if action_selected.sum() > 0:
                # adding 4 because the first 4 edges are location directional edges
                edgeids = self.workers_location[action_selected]*En + 4 + nonlocal_action
                targets = self.edges[edgeids]
                valid   = targets>=0
                if valid.sum() > 0:
                    all_targets = self.edges[self.workers_location*En + 4 + nonlocal_action]
                    w_valid = np.logical_and(action_selected, all_targets>=0)
                    self.workers_location[w_valid] = all_targets[w_valid]
                    self.workers_backward[w_valid] = 0
                    self.workers_motions [w_valid] = self.workers_motions[w_valid] + 1
                    moved[w_valid] = True

        # one-hot vectors for actions
        self.workers_prev_act[...] = 0
        self.workers_prev_act[np.arange(self.workers), action] = 1

        # only update observations for agents that moved
        if moved.sum() > 0: self.workers_obs[moved] = self._obs_at(self.workers_location[moved], self.workers_backward[moved], moved)

        # add some gaussian noise to agents that didn't move
        nmoved = np.logical_not(moved)
        if nmoved.sum() > 0: self.workers_obs[nmoved] += self.iid_noise(0, 0.01, nmoved.sum())

        goal_reached = self.workers_location == self.workers_goal
        time_expired = self.workers_timestep >= self.workers_ep_limit

        reward = goal_reached.astype(np.float32) - self.bump_penalty*bumped
        self.workers_done    = np.logical_or(goal_reached, time_expired)
        self.workers_reward += reward

        self.workers_success = goal_reached.astype(np.float32)

        self.worker_0_path.append(self.workers_location[0])

        return (self.workers_obs.copy(),
                self.workers_goal_obs.copy(),
                self.workers_prev_act.copy(),
                reward.copy(),
                self.workers_done.astype(np.float32).copy(),
                self.workers_location.copy(),
                self.workers_goal.copy())

    #--------------------------------------------------------------------------

    def _valid_actions(self, loc, backward):
        valid = [0,2] # you can always rotate
        En = 4+self.num_nonlocal_destinations
        if self.edges[loc*En + (backward+2)%4] >= 0: valid.append(1)
        nonlocal_edges_valid = [4+a for a in range(self.num_nonlocal_destinations) if self.edges[loc*En+4+a] >= 0]
        return valid + nonlocal_edges_valid

    #--------------------------------------------------------------------------

    def iid_noise(self, mean, sigma, size):
        if self._iid_index + size >= self._iid_noise.shape[0]: self._iid_index = 0
        noise = self._iid_noise[self._iid_index:self._iid_index+size]*sigma + mean
        self._iid_index += size
        return noise

    #--------------------------------------------------------------------------

    def ou_noise(self, w=None):
        if w is None: w = np.full((self.workers,), True, np.bool)
        self.workers_noise[w] = (1-self.noise_theta)*self.workers_noise[w] + np.random.normal(0,self.noise_sigma,w.sum())
        return self.workers_noise[w]

    #--------------------------------------------------------------------------

    def noise(self, gradients, w=None):
        if w is None: w = np.full((self.workers,), True, np.bool)
        cor_noise = self.ou_noise(w)[:,None]                  *gradients # correlated OU noise
        iid_noise = self.iid_noise(0,self.noise_sigma,w.sum())*gradients # uncorrelated noise
        return cor_noise + iid_noise

    #--------------------------------------------------------------------------

    def _obs_at(self, locations, backwards, w=None):
        if w is None: w = np.full((self.workers,), True, np.bool)
        varids = np.random.randint(0,self.obs_variations, len(locations))
        flatid = locations*self.obs_variations*4 + varids*4 + backwards
        return self.observations[flatid] + self.noise(self.obs_gradient[flatid], w)

    #--------------------------------------------------------------------------

    def _img_at(self, location, backward):
        id  = str(self.id_to_location[location])
        img = cv2.imdecode(self.graph["raw_nodes"][id]["frame"], cv2.IMREAD_COLOR)
        img = polar_to_cartesian(img, theta=self.workers_backward[0]*-np.pi/2, **self.graph["unwrapping_parameters"])
        return img

    #--------------------------------------------------------------------------

    def render(self, mode="human", wait="1", action_keys=None):
        i_o = self._img_at(self.workers_location[0], self.workers_backward[0])
        i_g = self._img_at(self.workers_goal[0], self.workers_g_back[0])
        return render(graph=self.graph, curriculum_level=self.curriculum_level,
                      start_location=self.workers_start[0],
                      curriculum_goals=self.curriculum_goals[self.curriculum_level, self.workers_start[0]*self.num_locations:(self.workers_start[0]+1)*self.num_locations],
                      id_to_location=self.id_to_location,
                      goal_location=self.workers_goal[0],
                      path_so_far=self.worker_0_path,
                      agent_location=self.workers_location[0],
                      agent_backward=self.workers_backward[0],
                      goal_backward=self.workers_g_back[0],
                      curriculum_levels=self.curriculum_levels,
                      timestep=self.workers_timestep[0],
                      time_limit=self.workers_ep_limit[0],
                      agent_features=self.workers_obs[0],
                      goal_features=self.workers_goal_obs[0],
                      agent_obs=i_o,
                      goal_obs=i_g,
                      mode=mode,
                      wait=wait,
                      action_keys=action_keys,
                      valid_actions=self._valid_actions(self.workers_location[0], self.workers_backward[0]))

