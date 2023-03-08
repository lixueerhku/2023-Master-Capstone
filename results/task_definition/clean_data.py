# Using readlines()
file = open('kp20k_multitask_prompt_1_no_stem.txt', 'r')
lines = file.readlines()
newlines=[]
  
count = 0
invalid = 0
# Strips the newline character
for line in lines:
    # print("line: "+line)
    count += 1
    newline=""
    if "keyphrases" in line:
        index = line.find("keyphrases")
        newline = line[index+12:].rstrip('\n').rstrip('.').rstrip()
    elif "topic" in line:
        index1 = line.find("topic")
        if "organization name" in line:
            index2 = line.find("organization name")
            newline = line[index1+7: index2].rstrip('\n').rstrip('.').rstrip()
        else:
            newline = line[index1+7:].rstrip('\n').rstrip('.').rstrip()
    else:
        newline=line.rstrip('\n').rstrip('.').rstrip()
    # print("New Line{}:{}".format(count, newline))
    newlines.append(newline)


    with open('experiments5_multitask_kp20k/kp20k_multitask_prompt_1_no_stem.txt', 'w') as f:
        for line in newlines:
            f.write(line)
            f.write('\n')