import csv 


# get data from dat file and merge with topo_file
def get_data(dat_file = './distance_matrix.dat', topo_file="./topo_file_list.txt"):
    
    with open(topo_file, 'r') as input_file:
        lines = input_file.readlines()
        topo_list = []
        for line in lines:
            val = line
            topo_list.append(val[:-2])
    print(",".join(str(e) for e in topo_list))
    newLines = [[",".join(str(e.split(".")[0]) for e in topo_list)]]

    with open(dat_file, 'r') as input_file:
        lines = input_file.readlines()
        for line in lines:
            newLine = line.split()
            newLines.append(newLine)

    with open('./distance_matrix.csv', 'w') as output_file:
        file_writer = csv.writer(output_file)
        file_writer.writerows(newLines)

get_data()


'''
with open('distance_matrix.dat', 'r') as input_file:
    lines = input_file.readlines()
    newLines = []
    for line in lines:
        newLine = line.strip('|').split()
        newLines.append(newLine)

with open('distance_matrix.csv', 'w') as output_file:
    file_writer = csv.writer(output_file)
    file_writer.writerows(newLines)
'''