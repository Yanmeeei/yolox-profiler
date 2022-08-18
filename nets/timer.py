import time
# import matplotlib.pyplot as plt


class Clock(object):
    def __init__(self):
        super().__init__()
        self.time_records = {}
        self.time_starts = {}

    def get_time(self, layername, time):
        if layername not in self.time_records:
            self.time_records[layername] = []
        self.time_records[layername].append(time)

    def tic(self, layername):
        self.time_starts[layername] = time.time()

    def toc(self, layername):
        diff = time.time() - self.time_starts[layername]
        if layername not in self.time_records:
            self.time_records[layername] = []
        self.time_records[layername].append(diff)

    def report(self, sample=False):
        # plt.rcParams.update({'font.size': 8})
        if sample:
            for key, value in self.time_records.items():
                # print("{:<15} {:<20}".format(key, value))
                print(key, end=' :: ')
                print(value, flush=True)

        layernames = []
        avg_times = []
        print("Average Time of Each Layer")
        for key, value in self.time_records.items():
            # value.pop(0)
            print("{:<15} {:<20,.4f} {:<5}".format(key, sum(value) / len(value) * 1000, " ms"))
            # print(f"{key} :: {sum(value) / len(value)}")
            # layernames.append(key)
            # avg_times.append(sum(value) / len(value))

        # fig = plt.figure()
        # # creating the bar plot
        # plt.bar(layernames, avg_times, width=0.4)
        #
        # plt.xlabel("layernames")
        # plt.xticks(rotation=45, ha="right")
        # plt.ylabel("avg_times (s)")
        # plt.savefig("layer-avg_time.png")

def parse_prof_table(prof_report):
    ret_list = []

    flip = False
    parsing_str = prof_report[0]
    parsing_idx = []
    for i in range(len(parsing_str)):
        if parsing_str[i] == '-':
            flip = True
        if flip and parsing_str[i] == ' ':
            parsing_idx.append(i)
            flip = False

    head_str_list = []
    parsing_str = prof_report[1]
    head_str = ""
    for i in range(len(parsing_str)):
        if i - 1 in parsing_idx:
            head_str_list.append(head_str.lstrip().rstrip())
            head_str = ""
        else:
            head_str += parsing_str[i:i + 1]

    ret_list.append(head_str_list)

    parsing_str_list = prof_report[3:-4]
    for parsing_str in parsing_str_list:
        head_str_list = []
        head_str = ""
        for i in range(len(parsing_str)):
            if i - 1 in parsing_idx:
                head_str_list.append(head_str.lstrip().rstrip())
                head_str = ""
            else:
                head_str += parsing_str[i:i + 1]
        ret_list.append(head_str_list)

    return ret_list