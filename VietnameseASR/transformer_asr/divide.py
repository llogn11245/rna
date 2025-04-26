import json

path = "refixed_valid.json"

with open(path, "r", encoding = 'utf-8') as f:
    json_data = json.load(f)

with open("refixed_valid_northern.json", 'r', encoding='utf-8') as f:
    easy_data = json.load(f)
with open("refixed_valid_central.json", 'r', encoding='utf-8') as f:
    medium_data = json.load(f)
with open("refixed_valid_southern.json", 'r', encoding='utf-8') as f:
    hard_data = json.load(f)
with open("refixed_valid_highlands.json", 'r', encoding='utf-8') as f:
    hard_data1 = json.load(f)
with open("refixed_valid_minorities.json", 'r', encoding='utf-8') as f:
    hard_data2 = json.load(f)

for i in json_data:
    if i not in easy_data and i not in medium_data and i not in hard_data and i not in hard_data1 and i not in hard_data2:
        print(json_data[i]['class'])

# easy_topic = {}
# hard_topic = {}
# medium_topic = {}

# for id, content in json_data.items():
#     if not content['class']:
#         continue
#     topic = content['class'][1]
#     if not topic.isdigit():
#         continue
#     topic = int(topic)
#     if topic == 0 or topic == 2:
#         easy_topic[id] = content
#     elif topic == 9 or topic == 4 or topic == 6:
#         hard_topic[id] = content
#     else:
#         medium_topic[id] = content

# # with open(path.replace(".json", "_medium.json"), 'w', encoding='utf-8') as f:
# #     json.dump(medium_topic, f, indent = 4, ensure_ascii=False)

# with open(path.replace(".json", "_easy.json"), 'w', encoding='utf-8') as f:
#     json.dump(easy_topic, f, indent = 4, ensure_ascii=False)

# with open(path.replace(".json", "_hard.json"), 'w', encoding='utf-8') as f:
#     json.dump(hard_topic, f, indent = 4, ensure_ascii=False)

# convert = {
#     0: "nothern",
#     1: "central",
#     2: "highlands",
#     3: "southern",
#     4: "minorities"
# }

# result = {}
# for k,v in convert.items():
#     result[v] = {}

# for id, content in json_data.items():
#     if not content['class']:
#         continue

#     region = content['class'][3]
#     if not region.isdigit():
#         continue
#     region = int(region)
#     result[convert[region]][id] = content

# for k, v in convert.items():
#     with open(path.replace(".json", f"_{v}.json"), "w", encoding='utf-8') as f:
#         json.dump(result[v], f, indent = 4, ensure_ascii=False)


    