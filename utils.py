import os

def get_numa_info():
    os.system('lscpu | grep "NUMA node" > lscpu.txt')
    
    # Using readlines()
    file1 = open('lscpu.txt', 'r')
    Lines = file1.readlines()

    if len(Lines) < 2:
        print("Error: can't get lscpu info.")
        os.system("rm -rf lscpu.txt")
        exit()
    
    snc_num = int(Lines[0].split(':')[1])
    print(f"SNC number: {snc_num}")

    # Strips the newline character
    numa_nodes = []
    for idx in range(len(Lines)):
        if idx == 0:
            continue
        numa_node = (Lines[idx].split(':')[1]).split(',')
        pyhsical_node_range=[int(x) for x in (str(numa_node[0]).split('-'))]
        logic_node_range=[int(x) for x in numa_node[1].split('-')]
        print(f"{Lines[idx]}  pyhsical_node_range={pyhsical_node_range}, logic_node_range={logic_node_range}")
        numa_nodes.append([pyhsical_node_range, logic_node_range])
    if snc_num != len(numa_nodes):
        print(f"Error: SNC number:{snc_num} != len(numa_nodes):{len(numa_nodes)} ")
        os.system("rm -rf lscpu.txt")
        exit()

    os.system("rm -rf lscpu.txt")
    return numa_nodes

def get_numa_nodes_num():
    numa_nodes = get_numa_info()
    return len(numa_nodes)

# numa_nodes = get_numa_info()
# print(f"numa_nodes={numa_nodes}")