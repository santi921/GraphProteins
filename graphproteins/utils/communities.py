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

    for ind, i in enumerate(unique_list):
        print(i + "\t\t" + str(count_list[ind]/sum_list))