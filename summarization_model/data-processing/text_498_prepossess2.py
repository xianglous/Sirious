a_file = open("/Users/shereen/Desktop/UM Courses/EECS 498/Sirious-main/summarization_model/file1126-2.csv", "r")
lines = a_file.readlines()
a_file.close()

new_file = open("/Users/shereen/Desktop/UM Courses/EECS 498/Sirious-main/summarization_model/file1126-2-new.csv", "w")
for line in lines:
    if line != "\"":
        line = line[1:]
        new_file.write(line)
new_file.close()

a_file = open("/Users/shereen/Desktop/UM Courses/EECS 498/Sirious-main/summarization_model/file1126-2-new.csv", "r")
lines = a_file.readlines()
a_file.close()

new_file = open("/Users/shereen/Desktop/UM Courses/EECS 498/Sirious-main/summarization_model/file1126-2-final.csv", "w")
for line in lines:
    if line != "\n":
        new_file.write(line)
new_file.close()