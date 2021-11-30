# with open(path_to_file) as f:
#     contents = f.readlines()
#     for line in contents:
#         if line[2]==":"


a_file = open("/Users/shereen/Desktop/UM Courses/EECS 498/Sirious-main/summarization_model/data/example/ted_does-schools-kill-creativity, cleaned.txt", "r")
lines = a_file.readlines()
a_file.close()

new_file = open("/Users/shereen/Desktop/UM Courses/EECS 498/Sirious-main/summarization_model/data/clean_data.txt", "w")
for line in lines:
    if line != "\n":
        if not line[0].isdigit():
            if line != "(Laughter)\n":
                new_file.write(line)

new_file.close()