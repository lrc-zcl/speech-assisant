import pyaudio
import numpy as np
from loguru import logger
import scipy.io.wavfile as wavfile


def record_audio_generator(duration=3, sample_rate=16000, channels=1, frames_per_buffer=1024):
    """
    :param duration: 录音时长
    :param sample_rate: 采样率
    :param channels: 通道数
    :param frames_per_buffer: 每帧缓冲区大小
    :return: 二进制数据流
    """
    record_count = 0
    while True:
        record_count = record_count + 1
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer
        )
        logger.info(f"\033[31mStart {record_count} epoch {duration}s recording\033[0m")
        frames = []
        for _ in range(0, int(sample_rate / frames_per_buffer * duration)):
            data = stream.read(frames_per_buffer)
            frames.append(data)
        logger.info(f"\033[31mrecord {record_count} epoch end\033[0m")

        stream.stop_stream()
        stream.close()
        audio.terminate()
        audio_bytes = b''.join(frames)
        yield audio_bytes


if __name__ == "__main__":
    count = 0
    for data in record_audio_generator():
        numpy_data = np.frombuffer(data, dtype=np.int16)
        normal_data = numpy_data / 32768.0
        wav_data_numpy = normal_data.astype(np.float32)
        wavfile.write(filename="./record.wav", rate=16000, data=wav_data_numpy)
        count += 1
        if count >= 2:
            break
    print("完成！")
