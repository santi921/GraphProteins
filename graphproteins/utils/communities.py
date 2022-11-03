def community_breakdown(coms_processed):
    unique_list = []
    count_list = []
    for i in coms_processed:
        if(not(i.split("_")[0] in unique_list)):
            unique_list.append(i.split("_")[0])
            count_list.append(0)
        ind = unique_list.index(i.split("_")[0])
        count_list[ind] += 1         
    
    sum_list = 0 
    for i in count_list: sum_list+=i 
    print("----------------------------------")
    print(sum_list)
    for ind, i in enumerate(unique_list):
        
        if i == '0': name = "Y"
        elif i == '1': name = "H"
        else: name = "C"

        print(name + "\t\t" + str(count_list[ind]/sum_list))