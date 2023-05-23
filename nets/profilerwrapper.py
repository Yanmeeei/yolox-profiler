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

        file = open("prof.csv", "w")
        writer = csv.writer(file)
        writer.writerow(["layer_name", "time", "output size", "mem consumption", "flops"])
        writer.writerow(["input", "nan", "%.4f" % (self.scale.scaleRec['input'] / (1024 ** 2)), "nan", "nan"])
        self.scale.scaleRec.pop('input')
        if self.mr.mem_cuda:
            for key, value in self.tt.time_records.items():
                writer.writerow([key,
                                 "%.4f" % (min(self.tt.time_records[key])),  # s
                                 "%.4f" % (self.scale.scaleRec[key] / (1024 ** 2)),  # MB
                                 "%.4f" % (sum(self.mr.mem_cuda[key]) / len(self.mr.mem_cuda[key])),
                                 0])
        else:
            for key, value in self.tt.time_records.items():
                writer.writerow([key,
                                 "%.4f" % (min(self.tt.time_records[key])),  # s
                                 "%.4f" % (self.scale.scaleRec[key] / (1024 ** 2)),  # MB
                                 "%.4f" % (sum(self.mr.mem_cpu[key]) / len(self.mr.mem_cpu[key])),
                                 0])
        file.close()
        file = open("dep.csv", "w")
        writer = csv.writer(file)
        writer.writerow(["src", "dst"])
        for key, value in self.scale.dependencyRec.items():
            for dest in value.dest_list:
                writer.writerow([value.src,
                                 dest])
        file.close()

