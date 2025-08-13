import re

def matchRegex(pattern: str, line: str):
    """Helper function for matching regex expressions."""
    match = re.search(pattern, line)
    if match:
        return tuple(match.groups())
    raise RuntimeError(f"regex matching failed on line {line}")

def parseOutput(slownode_file, benchmark, datatype):
    """Parses text output from slow_node.cc"""
    rank_times = {}
    rank_breakdowns = {}
    rank_to_node_map = {}
    rank_info_map = {}
    is_parsing=False
    with open(slownode_file, "r") as output:
        for line in output:
            if line.startswith("NodeInfo:"):
                # splits: ['NodeInfo:', hostname, world_rank, shared_rank]
                splits = line.split(" ")
                rank_info_map[int(splits[2])] = (splits[1], int(splits[3]))
            elif line.startswith(f"{benchmark}_{datatype}"):
                is_parsing = True
            elif is_parsing:
                if line.startswith("gather"):
                    # splits: ['gather', rank_info, total_time, 'breakdown', [times]]
                    splits = line.split(":")

                    # 1. Determine the Rank ID (and node name, if present)
                    raw_rank_info = splits[1].strip()
                    # raw_rank_info = 'rank_id (node)'
                    rank_info = re.findall(
                        r"(\d+)\s+\(([^)]+)\)",
                        raw_rank_info
                    )[0]
                    rank_id = int(rank_info[0])
                    node_name = rank_info[1]
                    rank_to_node_map[rank_id] = node_name

                    # 2. Get the total time for the current rank
                    total_time =  float(splits[2].strip())

                    # 3. Isolate the times for each iteration on the current rank
                    breakdown = splits[4].strip()
                    breakdown_list = [float(t) for t in breakdown.split(" ")]

                    # Populate rank data dicts
                    rank_times[rank_id] = total_time
                    rank_breakdowns[rank_id] = breakdown_list

                elif line.strip() == "":
                    is_parsing = False

    return rank_times, rank_breakdowns, rank_to_node_map, rank_info_map

def parseSensors(sensors_file):
    """
    Iterates through the sensors directory (given with -s on the command line)
    and identifies the temperature of each rank on that node.
    """
    node_temps = {}
    node_freqs = {}
    with open(sensors_file, 'r') as sensor_data:
        for line in sensor_data:
            if line.startswith("Node"):
                pattern = r"Node (\w+), Socket (\d+), Core (\d+): (\d+)(?:°C| C), (?:-|)(\d+) KHz"

                node_name,  \
                socket_str, \
                core_str,   \
                temp_str,   \
                freq_str = matchRegex(pattern, line)

                socket_id = int(socket_str)
                core_id = int(core_str)
                temp = float(temp_str)
                freq = int(freq_str)

                if node_name not in node_temps:
                    node_temps = {node_name: {}}
                    node_freqs = {node_name: {}}

                if socket_id not in node_temps[node_name]:
                    node_temps[node_name][socket_id] = {}
                    node_freqs[node_name][socket_id] = {}

                node_temps[node_name][socket_id][core_id] = temp
                node_freqs[node_name][socket_id][core_id] = freq

    return node_temps, node_freqs

def __isOpeningString(line: str):
    initial_exclude = "will be excluded from the hostfile" in line
    overflow_exclude = "the following nodes will also be omitted from the hostfile" in line
    return initial_exclude or overflow_exclude

def parseAnalysis(analysis_file):
    dropped_node_names = []
    is_reading_dropped_nodes = False

    with open(analysis_file, 'r') as analysis_output:
        for line in analysis_output:
            if __isOpeningString(line):
                is_reading_dropped_nodes = True
            elif is_reading_dropped_nodes and (line.strip() == "" or line.startswith("hostfile")):
                is_reading_dropped_nodes = False
            elif is_reading_dropped_nodes:
                node_name = line.strip().split(" ")[0]
                dropped_node_names.append(node_name)

    return dropped_node_names
