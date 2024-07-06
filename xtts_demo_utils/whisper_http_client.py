import io
import soundfile
import numpy as np
import wave
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import zhconv
from loguru import logger
from scipy.io import wavfile

padding_duration = 10
whisper_prompt = "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>"
client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
# wav_path = "/home/ai_triton_code/docs/examples/model_repository/chatglm3/white_6s_16K_16bit_1channel.wav"
wav_path = "/home/ai_triton_code/docs/examples/chatglm3-6b/rec-5000ms-16kbps-16000hz.wav"


def load_audio(wav_path):
    with open(wav_path, 'rb') as f:
        audio_bytes = f.read()
    waveform, samplerate = soundfile.read(file=io.BytesIO(audio_bytes), dtype='float32')
    assert samplerate == 16000, f"Only support 16k sample rate, but got {samplerate}"
    return waveform, samplerate


def convert_pcm_to_wav_and_get_info(audio_bytes):
    try:
        with io.BytesIO() as buf:  # 创建一个内存中的二进制数据流
            with wave.open(buf, 'wb') as w:  # 使用 wave 库创建一个 WAV 文件对象
                w.setnchannels(1)  # 设置声道数为 1
                w.setsampwidth(2)  # 设置采样宽度为 2 字节
                w.setframerate(16000)  # 设置采样率为 16000 Hz
                w.writeframes(audio_bytes)  # 将 PCM 格式的音频数据写入到 WAV 文件中

            buf.seek(0)  # 将内存指针移回到文件开头
            waveform, samplerate = soundfile.read(buf)  # 从二进制数据流中重新读取音频数据
        # 返回音频信息
        #wavfile.write("./test_output.wav", 16000, waveform)
        return waveform, samplerate

    except Exception as e:  # 捕获异常
        print("Failed to convert audio and retrieve information:", e)  # 打印异常信息
        return None, None


def whisper_http_fuc(wav_data):
    """
    :param wav_data: 输入的音频数据: BYTES
    :return: 识别结果: str
    """
    waveform, sample_rate = convert_pcm_to_wav_and_get_info(wav_data)
    padding_duration = 10
    whisper_prompt = "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>"
    client = httpclient.InferenceServerClient(url="192.168.1.19:8000", verbose=False)
    duration = int(len(waveform) / sample_rate)
    samples = np.zeros((1, padding_duration * sample_rate * ((duration // padding_duration) + 1)),
                       dtype=np.float32, )  # 填充数据
    inputs = [httpclient.InferInput("TEXT_PREFIX", [1, 1], 'BYTES'),
              httpclient.InferInput("WAV", samples.shape, np_to_triton_dtype(samples.dtype))]

    samples[0, : len(waveform)] = waveform  # 赋值
    input_data_numpy = np.array([whisper_prompt], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[0].set_data_from_numpy(input_data_numpy)
    inputs[1].set_data_from_numpy(samples)
    outputs = [httpclient.InferRequestedOutput("TRANSCRIPTS")]

    response = client.infer('whisper', inputs, outputs=outputs)
    decoding_results = response.as_numpy("TRANSCRIPTS")[0]
    if type(decoding_results) == np.ndarray:
        decoding_results = b" ".join(decoding_results).decode("utf-8")
    else:
        # For wenet
        decoding_results = decoding_results.decode("utf-8")
    print('ASR识别结果:', zhconv.convert(decoding_results, 'zh-hans'))
    logger.info("complete")
    return zhconv.convert(decoding_results, 'zh-hans')




if __name__ == "__main__":

    # waveform, sample_rate = load_audio(wav_path)
    with open("/home/ai_triton_code/docs/examples/chatglm3-6b/A2_0.wav", "rb") as f:
        w_data = f.read()
    waveform, sample_rate = convert_pcm_to_wav_and_get_info(w_data)
    duration = int(len(waveform) / sample_rate)
    print("useful_speech_data:", waveform, type(waveform), waveform.shape, '\n')
    samples = np.zeros((1, padding_duration * sample_rate * ((duration // padding_duration) + 1)),
                       dtype=np.float32, )  # 填充数据

    inputs = [httpclient.InferInput("TEXT_PREFIX", [1, 1], 'BYTES'),
              httpclient.InferInput("WAV", samples.shape, np_to_triton_dtype(samples.dtype))]

    samples[0, : len(waveform)] = waveform  # 赋值
    print("samples", samples, samples.shape, '\n')

    input_data_numpy = np.array([whisper_prompt], dtype=object)
    input_data_numpy = input_data_numpy.reshape((1, 1))
    inputs[0].set_data_from_numpy(input_data_numpy)
    inputs[1].set_data_from_numpy(samples)
    outputs = [httpclient.InferRequestedOutput("TRANSCRIPTS")]

    response = client.infer('whisper', inputs, outputs=outputs)
    decoding_results = response.as_numpy("TRANSCRIPTS")[0]
    if type(decoding_results) == np.ndarray:
        decoding_results = b" ".join(decoding_results).decode("utf-8")
    else:
        # For wenet
        decoding_results = decoding_results.decode("utf-8")
    print('识别结果:', zhconv.convert(decoding_results, 'zh-hans'))
    logger.info("complete")
