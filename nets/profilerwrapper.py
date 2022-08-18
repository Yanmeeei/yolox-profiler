import nets.timer as timer
import nets.memorizer as memorizer
import nets.datascale as datascale
import csv


class ProfilerWrapper(object):
    def __init__(self):
        self.mr = memorizer.MemRec()
        self.tt = timer.Clock()
        self.scale = datascale.DataScale()

    def report(self, sample=False):

        file = open("table_1.csv", "w")
        writer = csv.writer(file)
        writer.writerow(["layer_name", "time", "output size", "mem consumption", "flops"])
        writer.writerow(["user_input", "nan", "%.4f" % (self.scale.scaleRec['user_input'] / (1024 ** 2)), "nan", "nan"])
        self.scale.scaleRec.pop('user_input')
        if self.mr.mem_cuda:
            for key, value in self.tt.time_records.items():
                writer.writerow([key,
                                 "%.4f" % (min(self.tt.time_records[key]) * 1000),
                                 "%.4f" % (self.scale.scaleRec[key] / (1024 ** 2)),
                                 "%.4f" % (sum(self.mr.mem_cuda[key]) / len(self.mr.mem_cuda[key])),
                                 self.scale.flopRec[key]])
        else:
            for key, value in self.tt.time_records.items():
                writer.writerow([key,
                                 "%.4f" % (min(self.tt.time_records[key]) * 1000),
                                 "%.4f" % (self.scale.scaleRec[key] / (1024 ** 2)),
                                 "%.4f" % (sum(self.mr.mem_cpu[key]) / len(self.mr.mem_cpu[key])),
                                 self.scale.flopRec[key]])
        file.close()
        file = open("table_2.csv", "w")
        writer = csv.writer(file)
        writer.writerow(["tensor name", "src", "dst"])
        for key, value in self.scale.dependencyRec.items():
            for dest in value.dest_list:
                writer.writerow([key,
                                 value.src,
                                 dest])
        file.close()

