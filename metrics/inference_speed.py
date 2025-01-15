from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from collections import defaultdict


class InferenceSpeed:
    def __init__(self, get_audio_func, dataset_name="reach-vb/jenny_tts_dataset", num_samples=10, max_text_len=30):
        self.dataset = load_dataset(dataset_name)
        nltk.download('punkt_tab')
        self.__get_audio_func = get_audio_func
        self.__num_samples = num_samples
        self.__max_text_len = max_text_len

    def __calculate_generation_time(self, text):
        start_time = time.time()
        self.__get_audio_func(text)
        end_time = time.time()

        return end_time - start_time

    def plot_inference_speed(self):
        average_speed = defaultdict(lambda: (0, 0))

        for text in tqdm(self.dataset['train']['transcription_normalised']):
            len_text = len(word_tokenize(text))
            if len_text >= self.__max_text_len or average_speed[len_text][1] > self.__num_samples:
                continue

            t = self.__calculate_generation_time(text)
            average, num = average_speed[len_text]
            average_speed[len_text] = (
                (average * num + t) / (num + 1), num + 1)

        x, y = zip(*sorted(average_speed.items()))

        plt.plot(x, [item[0] for item in y])
        plt.title("Inference speed")
        plt.xlabel("Tokens count")
        plt.ylabel("Inference time")
        plt.grid(True)
        plt.show()
