from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import numpy as np
from scipy.stats import norm


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

    def plot_inference_speed(self, alpha=0.05):
        average_speed = defaultdict(lambda: (0, 0, 0))

        for text in tqdm(self.dataset['train']['transcription_normalised']):
            len_text = len(word_tokenize(text))
            if len_text >= self.__max_text_len or average_speed[len_text][1] > self.__num_samples:
                continue

            t = self.__calculate_generation_time(text)
            average, count, variance = average_speed[len_text]

            count += 1
            new_average = (average * (count - 1) + t) / count
            new_variance = ((variance * (count - 1)) +
                            (t - new_average) * (t - average)) / count

            average_speed[len_text] = (new_average, count, new_variance)

        x, y = zip(*sorted(average_speed.items()))
        means = [item[0] for item in y]
        counts = [item[1] for item in y]
        variances = [item[2] for item in y]

        confidence_intervals_lower = []
        confidence_intervals_upper = []
        for i in range(len(means)):
            std_error = np.sqrt(variances[i] / counts[i])
            z_value = norm.ppf(1 - alpha / 2)
            ci = z_value * std_error
            confidence_intervals_lower.append(means[i] - ci * (counts[i] > 1))
            confidence_intervals_upper.append(means[i] + ci * (counts[i] > 1))

        plt.plot(x, means, label="Average Inference Time", color="tab:orange")
        plt.fill_between(x, confidence_intervals_lower,
                         confidence_intervals_upper, color="tab:orange", alpha=0.3)

        plt.title("Inference Speed with Confidence Intervals")
        plt.xlabel("Tokens Count")
        plt.ylabel("Inference Time (seconds)")
        plt.grid(True)
        plt.show()
