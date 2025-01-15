import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import numpy as np
from tqdm import tqdm
from jiwer import wer
import soundfile as sf
import scipy.stats as sts
from config import TMP_PATH


class WER:
    def __init__(self, topics=None, llm_name="gpt2", dataset_size=100, asr_model_name="openai/whisper-large-v3-turbo"):
        self.llm_name = llm_name
        if topics is None:
            self.topics = [
                "technology",
                "travel",
                "science",
                "art",
                "space",
                "sports",
                "culture",
                "history",
                "health",
                "business",
            ]
        self.dataset_size = dataset_size
        self.asr_model_name = asr_model_name
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.__temp_audio_path = os.path.join(TMP_PATH, "temp_audio.wav")
        self.__wer_results = None

    def __load_llm_model_and_tokenizer(self):
        self.__tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        self.__model = AutoModelForCausalLM.from_pretrained(self.llm_name)

    def __generate_text_for_topic(self, topic, max_length=50):
        prompt = f"Write something interesting about {topic}: "
        inputs = self.__tokenizer(prompt, return_tensors="pt")
        outputs = self.__model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
        generated_text = self.__tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        clean_text = generated_text[len(prompt):].strip()
        return clean_text

    def __generate_text_dataset(self):
        self.__load_llm_model_and_tokenizer()

        selected_topics = np.random.choice(
            self.topics, self.dataset_size, replace=True
        )

        self.__text_dataset = []
        for topic in tqdm(selected_topics):
            text = self.__generate_text_for_topic(topic)
            self.__text_dataset.append(text)

    def __make_asr_pipeline(self):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_asr = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.asr_model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )

        model_asr.to(self.device)

        processor = AutoProcessor.from_pretrained(self.asr_model_name)

        self.__asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_asr,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def __count_wer(self, sr, get_audio_func):
        self.__generate_text_dataset()
        self.__make_asr_pipeline()
        self.__wer_results = []
        for original_text in tqdm(self.__text_dataset):
            audio_arr = get_audio_func(original_text)

            sf.write(self.__temp_audio_path, audio_arr, sr)

            recognized_text = self.__asr_pipeline(
                self.__temp_audio_path)["text"]

            error_rate = wer(original_text, recognized_text)

            self.__wer_results.append(error_rate)

        self.__wer_results = np.array(self.__wer_results)
        os.remove(self.__temp_audio_path)

    def count_wer(self, sr, get_audio_func):
        if self.__wer_results is None:
            self.__count_wer(sr, get_audio_func)
        return self.__wer_results.mean()

    def count_wer_conf_interval(self, sr, get_audio_func, alpha=0.05):
        if self.__wer_results is None:
            self.__count_wer(sr, get_audio_func)
        mean = self.__wer_results.mean()

        z = sts.norm(0, 1)

        deviation = z.ppf(1 - alpha / 2) * \
            self.__wer_results.std() / np.sqrt(self.dataset_size)
        return (mean - deviation, mean + deviation)
