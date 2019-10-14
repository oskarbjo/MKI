
filename = r"C:\Users\objorkqv\cernbox\Documents\Python\MKI workspace\MKI powerloss\injection.tfs"
text_file = open(filename, "r")
lines = text_file.read().split(',')
names = []
values = []

for i in range(len(lines)):
    data = lines[i].split(': ')
    names.append(data[0])
    names[i] = names[i].replace('\'','')
    names[i] = names[i].replace(' ','')
    values.append(data[1])
    try:
        
    except:
        values[i] = []

print(' ')