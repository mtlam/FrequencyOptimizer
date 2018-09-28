infile = open("psr_info.txt", "r")

lines = infile.readlines()
lines = [line.strip() for line in lines]

keys = lines[0].split()

print("nanograv_psrs = {")

for line in lines[1:]:
    sline = line.split()
    print("    \"%s\" : {"%sline[0])
    for ii,value in enumerate(sline):
        if value == "None":
            print("        \"%s\" : None,"%(keys[ii]))            
        elif ii == 0 or ii == 8:
            print("        \"%s\" : \"%s\","%(keys[ii],value))
        else:
            print("        \"%s\" : %f,"%(keys[ii],float(value)))               
    print("    },")
print("}")

