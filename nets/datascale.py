import torch
import sys


class DpdtRec:
    def __init__(self, src_name, src):
        self.src_name = src_name
        self.src = src
        self.dest_list = []

    def update(self, dest):
        self.dest_list.append(dest)


class DataScale(object):
    def __init__(self):
        self.scaleRec = {}
        self.dependencyRec = {}
        self.flopRec = {}

    def weight(self, tensor_src, data):
        if tensor_src not in self.scaleRec:
            # self.scaleRec[tensor_src] = data.numel() * data.element_size() / 4
            # self.scaleRec[tensor_src] = sys.getsizeof(data.storage())
            if data.is_sparse:
                self.scaleRec[tensor_src] = data.indices.numel() * data.indices.element_size() \
                              + data.values.numel() * data.indices.element_size()
            else:
                self.scaleRec[tensor_src] = data.numel() * data.element_size()

    def dependency_check(self, tensor_name, src, dest):
        src_name = src + "_" + tensor_name
        if src_name not in self.dependencyRec:
            self.dependencyRec[src_name] = DpdtRec(src_name, src)
        self.dependencyRec[src_name].update(dest)

    def flop_count(self, layer_name, num_flops):
        if layer_name not in self.flopRec:
            self.flopRec[layer_name] = num_flops

    def report(self):
        print("\nData size of each tensor: \n")
        print("{:<15} {:<20}".format('Name', 'Size'))
        print("================================================\n")
        for key, value in self.scaleRec.items():
            print("{:<25} {:<20,.4f}".format(key, value/(1024**2)))

        print("\nData Dependency of each layer \n")
        print("{:<20} {:<15} {:<15}".format('Name', 'Source', "Destination"))
        print("================================================\n")
        for key, value in self.dependencyRec.items():
            for dest in value.dest_list:
                print("{:<40} {:<25} {:<15}".format(key, value.src, dest))