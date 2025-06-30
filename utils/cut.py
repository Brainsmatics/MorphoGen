import numpy as np
from collections import defaultdict
def read_swc(file_path):

    swc_data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = list(map(float, line.split()))
            node = {
                'n': int(parts[0]),
                'type': int(parts[1]),
                'x': parts[2],
                'y': parts[3],
                'z': parts[4],
                'radius': parts[5],
                'parent': int(parts[6])
            }
            swc_data.append(node)
    return swc_data


def analyze_topology(swc_data):
    children = defaultdict(list)
    node_dict = {}

    for node in swc_data:
        node_dict[node['n']] = node
        if node['parent'] != -1:
            children[node['parent']].append(node['n'])

    return children, node_dict


def calculate_path_length(node_id, node_dict, children):
    path = []
    current_id = node_id
    total_length = 0.0

    while current_id != -1:
        path.append(current_id)
        parent_id = node_dict[current_id]['parent']

        if parent_id in node_dict:
            p1 = np.array([node_dict[current_id]['x'],
                           node_dict[current_id]['y'],
                           node_dict[current_id]['z']])
            p2 = np.array([node_dict[parent_id]['x'],
                           node_dict[parent_id]['y'],
                           node_dict[parent_id]['z']])
            total_length += np.linalg.norm(p1 - p2)

        if len(children[parent_id]) > 1:
            break
        current_id = parent_id

    return total_length, path


def filter_short_branches(swc_data, length_threshold=20):
    children, node_dict = analyze_topology(swc_data)
    leaves = [node_id for node_id in node_dict if not children[node_id]]

    to_remove = set()
    for leaf in leaves:
        length, path = calculate_path_length(leaf, node_dict, children)
        if length < length_threshold:
            to_remove.update(path)

    preserved = [node for node in swc_data if node['n'] not in to_remove]

    id_map = {node['n']: i + 1 for i, node in enumerate(preserved)}
    id_map[-1] = -1

    new_swc = []
    for node in preserved:
        new_node = node.copy()
        new_node['n'] = id_map[node['n']]
        new_node['parent'] = id_map.get(node['parent'], -1)
        new_swc.append(new_node)

    return new_swc


def write_swc(data, output_path):

    with open(output_path, 'w') as f:
        f.write("## Filtered SWC file\n")
        f.write("# id type x y z radius parent\n")
        for node in data:
            line = (f"{node['id']} {node['type']} {node['x']:.3f} {node['y']:.3f} "
                    f"{node['z']:.3f} {node['radius']:.3f} {node['parent']}")
            f.write(line + '\n')


