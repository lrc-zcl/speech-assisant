import sys
import pyaudio
from loguru import logger
from stream_kws_ctc import KeyWordSpotter


def record_audio_generator(duration=3, sample_rate=16000, channels=1, frames_per_buffer=1024):
    """
    :param duration: 录音时长
    :param sample_rate: 采样率
    :param channels: 通道数
    :param frames_per_buffer: 每帧缓冲区大小
    :return: 字节流
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
        logger.info(f"Start {record_count} epoch recording")
        frames = []
        for _ in range(0, int(sample_rate / frames_per_buffer * duration)):
            data = stream.read(frames_per_buffer)
            frames.append(data)
        logger.info(f"Record {record_count} epoch end")

        stream.stop_stream()
        stream.close()
        audio.terminate()
        audio_bytes = b''.join(frames)
        yield audio_bytes


if __name__ == "__main__":
    kws = KeyWordSpotter(ckpt_path="./model/hixiaowen/avg_30.pt",
                         config_path="./model/hixiaowen/config.yaml",
                         token_path="./model/tokens.txt",
                         lexicon_path="./model/lexicon.txt",
                         threshold=0.02,
                         min_frames=5,
                         max_frames=250,
                         interval_frames=50,
                         score_beam=3,
                         path_beam=20,
                         gpu=1,
                         is_jit_model=False,
                         )
    kws.set_keywords("你好问问,嗨小问")
    interval = int(0.3 * 16000) * 2

    for epoch_speech_data in record_audio_generator():
        wav_data = epoch_speech_data
        frame = len(wav_data) // interval if len(wav_data) % interval == 0 else len(wav_data) // interval + 1  # 末尾处必取
        for i in range(frame):
            chunk_data = wav_data[i * interval:(i + 1) * interval]
            wav_up_result = kws.forward(chunk_data)
            logger.info(wav_up_result)
            if wav_up_result["score"] is not None:
                logger.info("Successfully detected wake-up words!")
                #break
                sys.exit(1)  # 直接终止程序