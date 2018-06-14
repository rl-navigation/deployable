from __future__ import print_function, division
import numpy as np, cv2, networkx as nx

#==============================================================================
# ENVIRONMENT RENDERER
#==============================================================================

_render_cache  = {}
_optimal_paths = None
def render(graph, id_to_location, goal_location, goal_backward, timestep,
        time_limit, agent_obs, goal_obs, goal_features, agent_features,
        agent_location=None, agent_backward=None, curriculum_levels=None,
        curriculum_level=None, start_location=None, curriculum_goals=None,
        mode="human", wait=1, action_probs=None, label_clusters=False,
        msg_color=None, path_so_far=None, msg=None, localization=None,
        goal_estimate=None, valid_actions=None, action_keys=None):

    global _render_cache, _optimal_paths

    W = H = 475; B = 20
    agent_color      = (0,255,0)
    goal_color       = (0,128,255)
    node_color       = (255,0,0)
    curriculum_color = (255,255,0)
    start_color      = (0,0,255)
    path_color       = (0,255,0)
    optimal_color    = (255,0,255)
    msg_color        = (0,0,255) if msg_color is None else msg_color

    # transform node positions to workspace coordinates
    def topos(node,offset=0):
        x,y = node["pose"][0:2]
        if len(node["origin"]) == 3: dx,dy,dtheta = node["origin"]
        else: dx, dy = node["origin"]; dtheta = 0
        xr = x*np.cos(dtheta) + y*np.sin(dtheta)
        yr = x*np.cos(dtheta+np.pi/2) + y*np.sin(dtheta+np.pi/2)
        x = xr + dx
        y = yr + dy
        if type(offset) is not int: offset = np.matmul(np.array([[np.cos(-dtheta), -np.sin(-dtheta)],[np.sin(-dtheta), np.cos(-dtheta)]]), offset)
        return np.array([x,y],np.float32) + offset

    # determine the size of the workspace
    poses = [topos(node) for node in graph["graph_nodes"].values()]
    poses = np.array(poses)
    xmn = poses[:,0].min(); xmx = poses[:,0].max(); xgp = xmx-xmn
    ymn = poses[:,1].min(); ymx = poses[:,1].max(); ygp = ymx-ymn
    if   ygp > xgp: xmn-=(ygp-xgp)/2; xmx+=(ygp-xgp)/2
    elif xgp > ygp: ymn-=(xgp-ygp)/2; ymx+=(xgp-ygp)/2
    def topix(x,y):
        return (int((x-xmn)/(xmx-xmn+1e-8)*(W-2*B)+B), H-int((y-ymn)/(ymx-ymn+1e-8)*(H-2*B)+B))

    render_key = (curriculum_level, start_location)
    if render_key in _render_cache:
        img = _render_cache[render_key].copy()

    else:
        img = np.full((H,W,3), 255, np.uint8)

        cluster_centers = {}

        # draw edges first
        for node_id,node in graph["graph_nodes"].items():
            for target_id,edge_type,edge_direction in node["edges"]:
                p1 = topix(*topos(node))
                p2 = topix(*topos(graph["graph_nodes"][str(target_id)]))
                cv2.line(img, p1, p2, (0,0,0), 2)

        # draw nodes
        for node_id,node in graph["graph_nodes"].items():
            pos = topix(*topos(node))
            if node["cluster"] not in cluster_centers: cluster_centers[node["cluster"]] = []
            cluster_centers[node["cluster"]].append(list(pos))
            cv2.circle(img, pos, 5, node_color, -1)

        if curriculum_goals is not None:
            # draw curriculum-available goals
            for n_idx in curriculum_goals:
                node = graph["graph_nodes"][str(id_to_location[n_idx])]
                cv2.circle(img, topix(*topos(node)), 3, curriculum_color, -1)

        if label_clusters:
            # draw cluster ids
            for cluster in cluster_centers.keys():
                pos = tuple(list(map(int, np.mean(cluster_centers[cluster], axis=0))))
                cv2.putText(img, str(cluster), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        if start_location is not None:
            # draw optimal path
            if _optimal_paths is None:
                G = nx.Graph()
                for node_id,node in graph["graph_nodes"].items(): G.add_node(node_id)
                for node_id,node in graph["graph_nodes"].items():
                    for (target_id,edge_type,edge_direction) in node["edges"]:
                        G.add_edge(node_id, str(target_id))
                        G.add_edge(str(target_id), node_id)
                _optimal_paths = dict(nx.all_pairs_shortest_path(G))
            for node_id in _optimal_paths[id_to_location[start_location]][id_to_location[goal_location]]:
                cv2.circle(img, topix(*topos(graph["graph_nodes"][str(node_id)])), 3, optimal_color, -1)

            # draw start location
            node = graph["graph_nodes"][str(id_to_location[start_location])]
            cv2.circle(img, topix(*topos(node)), 4, start_color, -1)

        # draw little dot inside nodes
        for node_id,node in graph["graph_nodes"].items():
            pos = topix(*topos(node))
            cv2.circle(img, pos, 1, (0,0,0), -1)

        # draw legend
        cv2.putText(img, "agent",      (img.shape[1]-90,  15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, agent_color,      2)
        cv2.putText(img, "goal",       (img.shape[1]-90,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, goal_color,       2)
        cv2.putText(img, "nodes",      (img.shape[1]-90,  45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, node_color,       2)
        cv2.putText(img, "curriculum", (img.shape[1]-90,  60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, curriculum_color, 2)
        cv2.putText(img, "start",      (img.shape[1]-90,  75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, start_color,      2)
        cv2.putText(img, "path",       (img.shape[1]-90,  90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, path_color,       2)
        cv2.putText(img, "optimal",    (img.shape[1]-90, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, optimal_color,    2)

        # cache image so we don't have to do this every frame
        _render_cache[render_key] = img.copy()

    if path_so_far is not None:
        # render agent's path so far
        for loc in path_so_far:
            cv2.circle(img, topix(*topos(graph["graph_nodes"][str(id_to_location[loc])])), 1, path_color, -1)

    # render localization and goal estimates
    if localization is not None or goal_estimate is not None:
        lmx = localization .max() if localization  is not None else 1
        gmx = goal_estimate.max() if goal_estimate is not None else 1
        if localization  is not None: nl = len(localization)
        if goal_estimate is not None: nl = len(goal_estimate)
        for i in sorted(range(nl), key=lambda x: localization[x]+goal_estimate[x]):
            L = localization [i]/lmx if localization  is not None else 0
            G = goal_estimate[i]/gmx if goal_estimate is not None else 0
            color = (int(L*255), 0, int(G*255))
            pos = topix(*topos(graph["graph_nodes"][str(id_to_location[i])]))
            cv2.circle(img, pos, 5, color, -1)

    # draw goal
    goal_node = id_to_location[goal_location]
    cv2.circle(img, topix(*topos(graph["graph_nodes"][goal_node])), 10, goal_color, 2)

    if agent_location is not None:
        # draw agent
        agent_node = id_to_location[agent_location]
        cv2.circle(img, topix(*topos(graph["graph_nodes"][agent_node])), 8, agent_color, 2)
        vector = np.array([5*np.cos (graph["graph_nodes"][agent_node]["pose"][3]+agent_backward*-np.pi/2),
                           5*np.sin (graph["graph_nodes"][agent_node]["pose"][3]+agent_backward*-np.pi/2)])
        cv2.line  (img, topix(*topos(graph["graph_nodes"][agent_node])),
                        topix(*topos(graph["graph_nodes"][agent_node], vector)), agent_color, 3)

    if curriculum_level is not None or curriculum_levels is not None:
        # print curriculum level
        cv2.putText(img, "curriculum level {} / {}".format(curriculum_level, curriculum_levels), (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    # render time limit
    cv2.putText(img, "time limit {} / {}".format(int(timestep), int(time_limit)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    def getColorJet(v, vmin=0, vmax=1):
        c = [1., 1., 1.]  # white
        R, G, B = 0, 1, 2
        dv = 0
        if v < vmin: v = vmin
        if v > vmax: v = vmax
        dv = vmax - vmin
        if   v < (vmin + 0.25 * dv): c[R] = 0;  c[G] = 4 * (v - vmin) / dv
        elif v < (vmin + 0.5  * dv): c[R] = 0.; c[B] = 1 + 4 * (vmin + 0.25 * dv - v) / dv
        elif v < (vmin + 0.75 * dv): c[R] = 4 * (v - vmin - 0.5 * dv) / dv; c[B] = 0.
        else: c[G] = 1 + 4 * (vmin + 0.75 * dv - v) / dv; c[B] = 0.
        return tuple([int(ci*255) for ci in reversed(c)])

    # render action probabilities
    if action_probs is not None:
        num_nonlocal_destinations = len(action_probs)-3
        labels = ["L", "F", "R"] + list(map(str, range(num_nonlocal_destinations)))
        for i,p in enumerate(action_probs):
            color = getColorJet(p)
            bw = 20; bh = 10; w = 20; h = 50
            x = bw + 2*w*i
            y = img.shape[1]-bh
            cv2.rectangle(img, (x,y-20), (x+w, y-20-int(p*h+1)), color, -1)
            cv2.putText(img, labels[i], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(img, "policy", (20, img.shape[1]-20-50-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    # render observation and goal vectors
    def vec2img(v):
        if v is None: v = np.zeros((4096,), np.float32)
        v  = (v.flatten()-np.abs(v).min())/(np.abs(v).max()-np.abs(v).min()+1e-8)
        im = np.zeros((int(np.ceil(v.shape[0]**0.5)**2),3), np.uint8)
        im[:v.shape[0],0] =-np.clip(v,a_min=None,a_max=0)*255
        im[:v.shape[0],2] = np.clip(v,a_min=0,a_max=None)*255
        im = im.reshape(int(im.shape[0]**0.5), int(im.shape[0]**0.5), 3)
        im = cv2.resize(im, dsize=(W,H), interpolation=cv2.INTER_NEAREST)
        return im

    # render obs and goal vectors as a square image
    v_o = vec2img(agent_features)
    v_g = vec2img(goal_features)
    cv2.putText(v_o, "obs",  (20,35), cv2.FONT_HERSHEY_SIMPLEX, 1, agent_color, 4)
    cv2.putText(v_g, "goal", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 1, goal_color,  4)
    vob = np.hstack([v_o, v_g])
    vob = cv2.resize(vob, dsize=None, fx=img.shape[1]/vob.shape[1], fy=img.shape[1]/vob.shape[1])
    img = np.vstack([img, vob])

    # render observation and goal images
    i_g = goal_obs
    i_o = cv2.resize(agent_obs, dsize=(i_g.shape[1], i_g.shape[0]))
    cv2.putText(i_o, "obs image",  (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, agent_color, 2)
    cv2.putText(i_g, "goal image", (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, goal_color,  2)
    iob = np.vstack([i_o, i_g])

    # render user instruction
    if msg is not None:
        S = 1; T = 2; x = 20; y = 50
        ret, baseline = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, S, T)
        ygap = ret[1]+10
        for i,text in enumerate(msg.split("|")):
            ret, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, S, T)
            cv2.rectangle(iob, (x-10,y+i*ygap-ret[1]-10), (x+ret[0]+10,y+i*ygap+10), (0,0,0), -1)
        for i,text in enumerate(msg.split("|")):
            cv2.putText(iob, text, (x,y+i*ygap), cv2.FONT_HERSHEY_SIMPLEX, S, msg_color, T)

    # combine map and images
    iob = cv2.resize(iob, dsize=None, fx=img.shape[0]/iob.shape[0], fy=img.shape[0]/iob.shape[0])
    img = np.hstack([img, iob])

    if valid_actions is not None and action_keys is not None:
        S = 2; T = 4; x = 200
        key_string = "valid keys:   " + "   ".join([action_keys[action] for action in sorted(valid_actions)])
        ret, baseline = cv2.getTextSize(key_string, cv2.FONT_HERSHEY_SIMPLEX, S, T)
        cv2.rectangle(img, (x-10, img.shape[0]-ret[1]-50), (x+ret[0]+10, img.shape[0]), (0,0,0), -1)
        cv2.putText  (img, key_string, (x,img.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, S, (255,255,255), T)

    # 1080p aspect ratio
    dy  = 1080-img.shape[0]
    img = np.pad(img, ((dy//2,dy//2),(0,0),(0,0)), mode="constant", constant_values=0)
    img = cv2.resize(img, dsize=(1920,1080))

    # show in a window
    if mode == "human":
        cv2.imshow("GraphEnv", img)
        key = cv2.waitKey(wait)
        return key, img

    return -1, img

