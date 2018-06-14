from __future__ import print_function, division
import numpy as np, cv2, time, random, gym, networkx as nx, sys, torch, torchvision.models as models, math, torch.nn.functional as F, os
from skimage.transform import resize

sys.dont_write_bytecode = True
from status import StatusLine

#==============================================================================
# UNWRAPPING IMAGES
#==============================================================================

def polar_to_cartesian(img, crop_horz, crop_vert, crop_unwb, crop_unwt, theta):
    def unwrap(image, center, radius, angle_offset=0, total_angle=2*np.pi):
        nsamples = int(1.5*image.shape[0]*total_angle/(2*np.pi))
        samples  = np.linspace(0, total_angle, nsamples)[:-1] + angle_offset
        angle, magnitude = np.meshgrid(samples, list(reversed(np.arange(radius))))
        x = magnitude * np.cos(angle) + center[0]
        y = magnitude * np.sin(angle) + center[1]
        x, y = cv2.convertMaps(x.astype('float32'), y.astype('float32'), cv2.CV_32FC1)
        return cv2.remap(image, x, y, cv2.INTER_LINEAR).astype(np.uint8)

    def rotate(img, theta):
        degrees = theta*180.0/math.pi
        matrix  = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2),degrees,1)
        rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        return rotated

    img  = img[max(0,crop_vert[0]):min(img.shape[0],crop_vert[1]),
               max(0,crop_horz[0]):min(img.shape[1],crop_horz[1]),:]

    if abs(theta)>0.01: img = rotate(img, theta)

    imgs = []
    for a in [0,1*np.pi/2,2*np.pi/2,3*np.pi/2]:
        unw = unwrap(img, (img.shape[1]/2, img.shape[0]/2), img.shape[0]/2,
                   angle_offset=a+np.pi/4, total_angle=np.pi/2)
        unw = unw[np.clip( crop_unwt,0,unw.shape[0]-crop_unwb-2)
                 :np.clip(-crop_unwb,-(unw.shape[0]-crop_unwt),-1)]
        imgs.append(unw)

    img = np.flip(np.hstack(imgs), axis=1).copy()
    return img

#==============================================================================
# PRETRAINED VISION
#==============================================================================

def load_vision(vision_init):
    if vision_init.lower() == "imagenet" : model = models.resnet18(pretrained=True)
    if vision_init.lower() == "untrained": model = models.resnet18(pretrained=False)
    if vision_init.lower() == "places365":
        model = models.resnet18(pretrained=False, num_classes=365)
        model_file = 'pretrained/resnet18_places365.pth.tar'
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    return model

#------------------------------------------------------------------------------

def forward(model, x, stride):
    model.avgpool.stride = stride  # 1= pool with full overlap, 3 = pool with 1/2 overlap, 7 = pool with no overlap
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return x.view(x.shape[0], -1)

#------------------------------------------------------------------------------

def model_encode(model, imgs, stride):
    imgs     = np.array(imgs).astype(np.float32)/255
    imgs     = (imgs-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    imgs     = np.transpose(imgs, [0,3,1,2])
    batch    = torch.from_numpy(imgs).float().cuda()
    # finite differences for dataset augmentation in feature space
    eps      = 1e-4
    encoded  = forward(model, batch,     stride).data
    enc_high = forward(model, batch+eps, stride).data
    enc_low  = forward(model, batch-eps, stride).data
    gradient = (enc_high-enc_low) / (2*eps)
    return encoded.cpu().numpy(), gradient.cpu().numpy()

#------------------------------------------------------------------------------

def encode_batch(batch, model, graph, orientation_id, shift_id, stride):
    imgs = [b[1] for b in batch]
    encoded, gradient = model_encode(model, imgs, stride)
    for i in range(len(batch)):
        emb = encoded[i].reshape(-1)
        _embedding_statistics.update(emb)
        grd = gradient[i].reshape(-1)
        graph["raw_nodes"][batch[i][0]]["img"][orientation_id][shift_id] = emb
        graph["raw_nodes"][batch[i][0]]["grd"][orientation_id][shift_id] = grd
        graph["obs_size" ] = emb.shape

#------------------------------------------------------------------------------

class OnlineStatistics(object):
    def __init__(s): s.s = (0,0,0)
    def update(s,x): n, mu, M2 = s.s; n += 1; d1 = x-mu; mu += d1/n; d2 = x-mu; M2 += d1*d2; s.s = (n, mu, M2)
    def retrieve(s): return s.s[1], s.s[2]/(max(1,s.s[0]-1))
    def normalize(s,x): m,v = s.retrieve(); return (x-m)/(v**0.5)
_embedding_statistics = OnlineStatistics()

#==============================================================================
# LOADING GRAPH
#==============================================================================

def preprocess_images(graph, datafile, vision_batch, shifts, shift_radians, stride, vision_init, max_variations):
    if max_variations > 0: config = [shifts,shift_radians,stride,vision_init,max_variations]
    else: config = [shifts,shift_radians,stride,vision_init]

    if not os.path.exists(".cache"): os.mkdir(".cache")
    cachefile = ".cache/"+datafile.replace("/","_")+".unwrapped.{}.pytorch".format(".".join([str(v) for v in config]))
    if not os.path.exists(cachefile):
        with StatusLine("Unwrapping, resizing, and encoding images"):
            orientations = [0,-np.pi/2,-np.pi,-3*np.pi/2]
            shifts = range(-shifts//2, shifts//2+1)
            model  = load_vision(vision_init)
            unwrapping_parameters = graph["unwrapping_parameters"]
            for orientation_id,theta in enumerate(orientations): # forward and backward orientation
                for shift_id,shift in enumerate(shifts): # small stochastic rotations
                    batch = []
                    for node_id in sorted(graph["graph_nodes"].keys()):
                        node = graph["graph_nodes"][node_id]
                        node["observations"] = list(node["observations"])[:max_variations] if max_variations > 0 else sorted(node["observations"])
                        for obs_id in node["observations"]:
                            obs_node = graph["raw_nodes"][str(obs_id)]
                            if "img" not in obs_node:
                                obs_node["img"] = [[None for _ in shifts] for __ in orientations]
                                obs_node["grd"] = [[None for _ in shifts] for __ in orientations]

                            img = cv2.imdecode(obs_node["frame"], cv2.IMREAD_COLOR)
                            img = polar_to_cartesian(img, theta=theta+shift*shift_radians, **unwrapping_parameters) # rotate images by an amount in either direction
                            img = resize(img, (224,4*224), mode="constant")

                            # encode with a fixed vision network
                            batch.append((str(obs_id),img.copy()))
                            if len(batch) >= vision_batch:
                                encode_batch(batch, model, graph, orientation_id, shift_id, stride)
                                batch = []

                    if len(batch) > 0:
                        encode_batch(batch, model, graph, orientation_id, shift_id, stride)
                        batch = []

            graph["embedding_mu" ] = _embedding_statistics.retrieve()[0]
            graph["embedding_sig"] = _embedding_statistics.retrieve()[1]**0.5
            torch.save(graph, cachefile)
    else:
        with StatusLine("Loading cached preprocessed images from '{}'".format(cachefile)): graph = torch.load(cachefile)

    return graph

#------------------------------------------------------------------------------

def find_shortest_paths(graph):
    with StatusLine("Finding all shortest paths"):
        all_lengths = []; G = nx.Graph()
        for node_id,node in graph["graph_nodes"].items(): G.add_node(node_id)
        for node_id,node in graph["graph_nodes"].items():
            for (target_id,edge_type,edge_direction) in node["edges"]:
                G.add_edge(node_id, str(target_id))
                G.add_edge(str(target_id), node_id)
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        graph["path_lengths"] = {}
        for node_id in graph["graph_nodes"].keys():
            graph["path_lengths"][node_id] = {}
            for target_id in graph["graph_nodes"].keys():
                if target_id in lengths[node_id]: graph["path_lengths"][node_id][target_id] = lengths[node_id][target_id]
                else: graph["path_lengths"][node_id][target_id] = -1
                all_lengths.append(graph["path_lengths"][node_id][target_id])
        graph["all_lengths"] = all_lengths

#------------------------------------------------------------------------------

def load_graph_file(datafile):
    with StatusLine("Loading dataset"):
        graph = torch.load(datafile)
        for node in graph["graph_nodes"].values(): node["edges"] = list(node["edges"]) # convert from set to list
    print("Docstring:", graph["docstring"])
    return graph

#------------------------------------------------------------------------------

_graphs = {}
def load_graph(datafile, vision_batch, shifts, shift_radians, stride, vision_init, max_variations):
    config = (datafile,shifts,shift_radians,stride,vision_init,max_variations)

    if config not in _graphs:
        _graphs[config] = load_graph_file(datafile)
        _graphs[config] = preprocess_images(_graphs[config], datafile, vision_batch, shifts, shift_radians, stride, vision_init, max_variations)
        find_shortest_paths(_graphs[config])

    return _graphs[config]

