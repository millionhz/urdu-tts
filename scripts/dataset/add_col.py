import argparse

parser = argparse.ArgumentParser(description='Add column to csv.')
parser.add_argument('filename', help='the name of the input file')

args = parser.parse_args()
filename = args.filename

rows = []
with open(filename, 'r') as file:
    for line in file:
        uid, text = line.split("|")
        rows.append([uid, "", text])

with open("test.csv", 'w') as file:
    for row in rows:
        file.write("|".join(row))
