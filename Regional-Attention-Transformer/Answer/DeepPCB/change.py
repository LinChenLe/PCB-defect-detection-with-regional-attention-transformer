import re
import os
import pdb
path = "gt/"
a = 0
for file in os.listdir(path):
    print_txt = ""
    with open(path+file) as txts:
        for txt in txts:
            box = [x for x in re.findall("(.*) (.*) (.*) (.*) .*",txt)[0]]
            category = re.findall(".* .* .* .* (.*)",txt)[0]
            
            for word in box:
                print_txt += word+","
            print_txt += category+"\n"
    with open(path+file,'w') as txts:
        txts.write(print_txt)
            