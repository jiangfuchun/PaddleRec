import random
from paddlerec.core.reader import ReaderBase
from paddlerec.core.utils import envs

class Reader(ReaderBase):
    def init(self):
        self.sampling_rate = envs.get_global_env("hyper_parameters.sampling_rate")
    def generate_sample(self, line):
        def get_rand(low=0.0, high=1.0):
            return random.random()

        def pairwise_iterator():
            feature_names = ["query", "pos_cand", "neg_cand"]
            items = line.strip("\t\n").split(";")
            pos_num, neg_num = [int(i) for i in items[1].split(" ")]
            query = items[2].split(" ")
            for i in range(pos_num):
                for j in range(neg_num):
                    prob = get_rand()
                    if prob < self.sampling_rate:
                        pos_title_int = items[3 + i].split(" ")
                        neg_title_int = items[3 + pos_num + j].split(" ")
                        yield zip(feature_names, [query, pos_title_int, neg_title_int])

        return pairwise_iterator

