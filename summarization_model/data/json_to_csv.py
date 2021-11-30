# # Python program to convert
# # JSON file to CSV
 
# import json
# import csv


# with open('/Users/shereen/Desktop/UM Courses/EECS 498/Sirious-main/summarization_model/data/ted-summaries.json', 'r',  encoding="utf-8") as json_file:
#     with open("ted-summaries-output.json", "w",  encoding="utf-8") as output:
#         data = json.load(json_file)
#         json.dump(data, output, ensure_ascii=False)
        
#         with open('/Users/shereen/Desktop/UM Courses/EECS 498/Sirious-main/summarization_model/data/ted-summaries-output.json') as json_file_2:
#             data2 = json.load(json_file_2)

#             data_file = open('ted-summaries.csv', 'w')
#             csv_writer = csv.writer(data_file)

#             count = 0
#             for emp in data2:
#                 if count == 0:
#                     header = emp.keys()
#                     csv_writer.writerow(header)
#                     count += 1
#                 csv_writer.writerow(emp.values())

#             data_file.close()


import json
import csv
 
with open('/Users/shereen/Desktop/UM Courses/EECS 498/Sirious/summarization_model/data/ted-summaries-newest.json') as json_file:
    jsondata = json.load(json_file)
 
data_file = open('ted-summaries-newest.csv', 'w')
csv_writer = csv.writer(data_file)

count = 0
for data in jsondata:
    if count == 0:
        header = data.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(data.values())
 
data_file.close()