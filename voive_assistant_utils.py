import time
import numpy as np
import pyaudio
from stream_kws_ctc import KeyWordSpotter
from loguru import logger
from ppasr.infer_utils.vad_predictor import VADPredictor
from xtts_demo_utils.whisper_http_client import whisper_http_fuc
from speaker_confirmation import test_infer
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


def speaker_confirm_signal3s_data(ref_wav_data, wav_data):
    """
    说话人确认函数
    :param ref_wav_data: 参考的音频数据: path or data(numpy[shepe = (xxx,)])
    :param wav_data:
    :return:
    """
    global speaker_confirm_state
    numpy_data = np.frombuffer(wav_data, dtype=np.int16)
    normal_data = numpy_data / 32768.0
    wav_data_numpy = normal_data.astype(np.float32)
    # while True:
    # 开始对每一帧数据进行vad检测
    vad_predictor = VADPredictor()
    speech_timestamps = vad_predictor.get_speech_timestamps(wav_data_numpy, 16000)
    useful_speech_data = []
    for speech_timestamp in speech_timestamps:
        start, end = speech_timestamp['start'], speech_timestamp['end']
        corp_wav = wav_data_numpy[start: end]
        useful_speech_data.extend(corp_wav)  # 获取到有用数据
    if len(useful_speech_data) != 0:
        model_id = "iic/speech_eres2net_sv_zh-cn_16k-common"
        wavs = [ref_wav_data, useful_speech_data]
        pre_model = r"D:\wenetkws\speaker_confirmation\model\speech_eres2net_sv_zh-cn_16k-common\pretrained_eres2net_aug.ckpt"
        speaker_score = test_infer.speaker_confirm_main(model_id, wavs, pre_model)
        logger.info(f"当前得分是{speaker_score}")
    else:
        logger.info("有效数据是空，请大声说话进行说话人身份确认！")
        time.sleep(1)
        return None
    assert speaker_score >= 0.3, "说话人身份确认失败，相似度小于0.3!"
    logger.info("身份确认成功，请继续下面操作！")
    speaker_confirm_state = True
    return wav_data


def kws_fuc(wav_data):
    """
    唤醒词识别功能
    :param wav_data: 输入的语音语音数据: bytes
    :return: 检测是否成功
    """
    interval = int(0.3 * 16000) * 2
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
    frames = len(wav_data) // interval if len(wav_data) % interval == 0 else len(wav_data) // interval + 1  # 末尾处必取
    for i in range(frames):
        chunk_data = wav_data[i * interval:(i + 1) * interval]
        wav_up_result = kws.forward(chunk_data)
        logger.info(f"开始检测第{i}帧: {wav_up_result}")
        if wav_up_result["score"] is not None:
            return True


def asr_fuc(wav_data):
    result = whisper_http_fuc(wav_data)
    return result
