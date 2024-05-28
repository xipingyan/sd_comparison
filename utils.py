import os

def print_start_flag(str):
    print("\n\n*******************************************************************************************************")
    print(f"== Start {str}")
    print("*******************************************************************************************************")

def get_numa_info():
    tmp_fn = "lscpu_" + str(os.getpid()) + ".txt"
    print(f"tmp_fn={tmp_fn}")
    os.system('lscpu | grep "NUMA node" > ' + tmp_fn)
    
    # Using readlines()
    os.getpid()
    file1 = open(tmp_fn, 'r')
    Lines = file1.readlines()

    if len(Lines) < 2:
        print("Error: can't get lscpu info.")
        os.system("rm -rf " + tmp_fn)
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
        os.system("rm -rf " + tmp_fn)
        exit()

    os.system("rm -rf " + tmp_fn)
    return numa_nodes

def get_numa_nodes_num():
    numa_nodes = get_numa_info()
    return len(numa_nodes)

numa_nodes = get_numa_info()
print(f"numa_nodes={numa_nodes}")