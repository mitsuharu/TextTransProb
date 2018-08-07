import os
import math
import pickle
from collections import defaultdict


class TransProb:
    """
    文字列の文字遷移パターンを学習し，生成遷移確率を計算する
    """

    mat = None
    non_pattern_prob = 0
    model_file = ""
    ngram = 1

    def __init__(self, model_file=None):
        if model_file is not None:
            self.load_model(model_file=model_file)

    def read_prob_mat(self, key0: str, key1: str):
        tmp_d = self.mat
        tmp_v = self.non_pattern_prob
        if self.mat:
            for key in [key0, key1]:
                tmp_v = tmp_d.get(key, self.non_pattern_prob)
                if isinstance(tmp_v, dict):
                    tmp_d = tmp_v
                else:
                    break
        return tmp_v

    def save_model(self, save_file: str = None):
        save_model_file = save_file
        if save_file is None:
            save_model_file = self.model_file

        if save_model_file is None or len(save_model_file) == 0:
            print("[error] save_file {} is nothing".format(save_file))
            return

        os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
        pickle.dump({"mat": self.mat,
                     "non_pattern_prob": self.non_pattern_prob,
                     "ngram": self.ngram},
                    open(save_model_file, "wb"))

    def load_model(self, model_file: str):
        if os.path.exists(model_file) is False:
            print("[error] model_file {} is not found".format(model_file))
            return
        try:
            model_data = pickle.load(open(model_file, "rb"))
            self.model_file = model_file
            self.mat = model_data["mat"]
            self.non_pattern_prob = model_data["non_pattern_prob"]
            self.ngram = model_data["ngram"]
        except Exception as e:
            print("e:", e)

    def train(self,
              training_file: str,
              save_file: str = "model.pki",
              ngram: int = 1):

        if os.path.exists(training_file) is False:
            print("[error] training_file {} is not found.".format(training_file))

        transition_mat = defaultdict(lambda: defaultdict(int))

        for line in open(training_file):
            tmp_line = line.rstrip("\r\n")
            for a, b in self.sublines_for_ngram(tmp_line, n=ngram):
                transition_mat[a][b] += 1

        # max normalization constant
        max_nc = 0
        for k, v in transition_mat.items():
            s = float(sum(v.values()))
            if max_nc < s:
                max_nc = s
        if max_nc == 0:
            max_nc = 50

        # to reduce data size, it calculates prob of patterns not in training data
        non_pattern_prob = math.log(1 / (max_nc * 2))

        # normalize
        for key0, dict0 in transition_mat.items():
            total = float(sum(dict0.values()))
            for key1, value1 in dict0.items():
                if value1 > 0:
                    dict0[key1] = math.log(float(value1)/ total)

        self.mat = dict(transition_mat)
        self.non_pattern_prob = non_pattern_prob
        self.ngram = ngram
        self.model_file = save_file
        self.save_model()

    def calc_prob(self, text: str):
        log_prob = 0.0
        trans_ct = 0
        for a, b in self.sublines_for_ngram(text):
            p = self.read_prob_mat(a, b)
            log_prob += p
            trans_ct += 1
        prob = math.exp(log_prob / (trans_ct or 1))
        return prob

    def sublines_for_ngram(self, input_line: str, n=None):

        def subline_with_upper_limit(line, sub_index, sub_length):
            subline = line[sub_index:]
            if sub_length <= len(subline):
                subline = subline[:sub_length]
            return subline

        if n is None:
            n = self.ngram

        terminal_char = "\0"

        if terminal_char in input_line:
            line = input_line
        else:
            line = input_line + terminal_char

        char_list = [c for c in line]
        for index, c in enumerate(char_list):
            if c == terminal_char:
                continue

            sublines0 = subline_with_upper_limit(line, index, n)

            next_index = index + n
            if terminal_char in sublines0:
                sublines0 = sublines0.replace(terminal_char, "")
                next_index = index + len(sublines0)

            sublines1 = subline_with_upper_limit(line, next_index, n)

            yield sublines0, sublines1


if __name__ == "__main__":
    print("[demo] TransProb")

    training_file = "./samples/en_words.txt"
    model_file = "./samples/model.pki"

    tp = TransProb()
    tp.train(training_file=training_file, save_file=model_file)
    print("p =", tp.calc_prob("pen"))
    print("p =", tp.calc_prob("aaa"))

    tp2 = TransProb(model_file=model_file)
    print("p =", tp2.calc_prob("pen"))
    print("p =", tp2.calc_prob("aaa"))
