import librosa
import numpy as np
from tqdm import tqdm
import scipy.stats as sts


class AudioSimilarity:
    def __init__(self, dataset, get_audio_func, dataset_size=100):
        self.__get_audio_func = get_audio_func
        self.dataset = dataset
        self.metrics = None
        self.dataset_size = dataset_size

    def __count_similarity(self):
        self.metrics = []

        data = self.dataset['train']
        indices = np.random.choice(len(data), size=self.dataset_size).tolist()

        for elem_number in tqdm(indices):

            audio_orig = data[elem_number]['audio']
            text = data[elem_number]['transcription_normalised']

            mel_original = librosa.feature.melspectrogram(
                y=audio_orig["array"], sr=audio_orig["sampling_rate"])
            mel_synthesis = self.__get_audio_func(text)

            self.metrics.append(librosa.segment.cross_similarity(
                mel_original, mel_synthesis).mean())
        self.metrics = np.array(self.metrics)

    def count_similarity(self):
        if self.metrics is None:
            self.__count_similarity()
        return self.metrics.mean()

    def count_similarity_conf_interval(self, alpha=0.05):
        if self.metrics is None:
            self.__count_similarity()

        mean = self.metrics.mean()

        z = sts.norm(0, 1)

        deviation = z.ppf(1 - alpha / 2) * \
            self.metrics.std() / np.sqrt(self.dataset_size)
        return (mean - deviation, mean + deviation)
