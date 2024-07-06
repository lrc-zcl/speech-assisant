import numpy as np
from scipy.io import wavfile
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import io
import soundfile
import time
import zhconv


def load_audio(wav_path):
    """
    :param wav_path: 可读wav文件地址
    :return: 音频numpy数组，采样率
    """
    with open(wav_path, 'rb') as f:
        audio_bytes = f.read()
    waveform, samplerate = soundfile.read(file=io.BytesIO(audio_bytes), dtype='float32')
    assert samplerate == 16000, f"Only support 16k sample rate, but got {samplerate}"
    return waveform, samplerate


def prepare_tensor(name, input):  # 建立client输入 numpy转成triton tensor
    t = httpclient.InferInput(name, input.shape,
                              np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input, binary_data=False)
    return t


def asr_audio(client, whisper_prompt, samples):
    """
    实现语音转写任务
    :param client:  建立客户端
    :param whisper_prompt: 转写任务前缀词
    :param samples: 音频数据: numpy数组
    :return: 转写文本
    """
    inputs = [
        httpclient.InferInput("TEXT_PREFIX", [1, 1], 'BYTES'),
        httpclient.InferInput("WAV", samples.shape, np_to_triton_dtype(samples.dtype))
    ]
    input_data_numpy = np.array([whisper_prompt], dtype=object).reshape((1, 1))
    inputs[0].set_data_from_numpy(input_data_numpy)
    inputs[1].set_data_from_numpy(samples)
    outputs = [httpclient.InferRequestedOutput("TRANSCRIPTS")]
    response = client.infer('whisper', inputs, outputs=outputs)
    decoding_results = response.as_numpy("TRANSCRIPTS")[0]
    if type(decoding_results) == np.ndarray:
        decoding_results = b" ".join(decoding_results).decode("utf-8")
    else:
        decoding_results = decoding_results.decode("utf-8")
    return zhconv.convert(decoding_results, 'zh-hans')


def chatglm_fuc(prompt, history_data_list, history_data_len):
    """
    实现大模型对话任务
    :param client: 建立客户端
    :param prompt: 输入文本: str
    :param history_data_list: 历史输入数据: list
    :param history_data_len: 历史输入长度: str
    :return:
    """
    client = httpclient.InferenceServerClient(url="192.168.1.19:8000", verbose=False)
    inputs = [
        prepare_tensor("prompt", np.array([[prompt]], dtype=object)),
        prepare_tensor("history", np.array([history_data_list], dtype=object)),
        prepare_tensor("temperature", np.array([["0.3"]], dtype=object)),
        prepare_tensor("max_token", np.array([["500"]], dtype=object)),
        prepare_tensor("history_len", np.array([[history_data_len]], dtype=object))
    ]
    response_output = httpclient.InferRequestedOutput('response', binary_data=False)
    history_output = httpclient.InferRequestedOutput('history', binary_data=False)
    output = [response_output, history_output]
    result = client.infer("chatglm3", inputs=inputs, outputs=output)
    print("大模型对话结果:",str(result.as_numpy("response")))
    return str(result.as_numpy('response'))


def xtts_fuc(text, language):
    """
    实现语音合成任务
    :param client: 建立客户端
    :param text: 输入文本数据: str
    :param language: 输入转录语言类型: str
    :return: 转录音频保存路径
    """
    client = httpclient.InferenceServerClient(url="192.168.1.19:8000", verbose=False)
    inputs = [
        prepare_tensor("text", np.array([[text]], dtype=object)),
        prepare_tensor("language", np.array([[language]], dtype=object))
    ]
    output_data = httpclient.InferRequestedOutput('output_data', binary_data=True)
    output = [output_data]
    result = client.infer("xtts_triton", inputs=inputs, outputs=output)
    return result.as_numpy('output_data')


if __name__ == "__main__":
    # current_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    # print('此时', current_time)
    #
    client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
    # wav_path = "/home/ai_triton_code/docs/examples/model_repository/chatglm3/pingjie.wav"
    # whisper_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    # # whisper_prompt = "zh"
    # t0 = time.time()
    #
    # waveform, sample_rate = load_audio(wav_path)
    # duration = int(len(waveform) / sample_rate)
    # padding_duration = 10
    # samples = np.zeros((1, padding_duration * sample_rate * ((duration // padding_duration) + 1)),
    #                    dtype=np.float32)
    # samples[0, : len(waveform)] = waveform  # 数据赋值
    #
    # decoding_results = asr_audio(client, whisper_prompt, samples)
    # print(f"识别结果是: {decoding_results}")
    # print(f"ASR识别所用时间: ", time.time() - t0, '\n')
    #
    # t2 = time.time()
    # history_data_list = []
    # history_data_len = "0"
    # llm_response = chatglm_fuc(client, decoding_results, history_data_list, history_data_len)
    # print("语音助手:", llm_response)
    # print("chatglm所用时间:", time.time() - t2, '\n')

    t3 = time.time()
    llm_response = "你好问问"
    tts_output_result = xtts_fuc(client, llm_response, "zh-cn")
    print("tts_output_data:", tts_output_result)
    print(type(tts_output_result), tts_output_result.shape)
    print("XTTS所用时间", time.time() - t3, '\n')
    print("complete")

    current_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    out_put_path = '/home/ai_triton_code/docs/examples/model_repository/chatglm3/' + current_time + '.wav'
    wavfile.write(out_put_path, 16000, tts_output_result)
