import time
from scipy.io import wavfile
from loguru import logger
from xtts_demo_utils.all_speech_triton_model_client import chatglm_fuc, xtts_fuc
from voive_assistant_utils import record_audio_generator, speaker_confirm_signal3s_data, kws_fuc, asr_fuc

if __name__ == "__main__":
    ref_wav_data = r"C:/Users/36974/Desktop/read/lrc.wav"
    asr_result = ""
    logger.info(f"\033[31m注意！ 2s后录音开始！\033[0m")
    time.sleep(2)
    speaker_confirm_state = False
    for epoch_speech_data in record_audio_generator():
        wav_data = epoch_speech_data

        if not speaker_confirm_state:
            speaker_result = speaker_confirm_signal3s_data(ref_wav_data, wav_data)

            if speaker_result is None:
                # logger.info("有用数据是空，重新录音")
                continue
            else:  # speaker_confirm_state ==True
                kws_result = kws_fuc(wav_data)

                if kws_result is None:
                    logger.info(f"\033[31m唤醒失败！1s后重新唤醒！\033[0m")
                    time.sleep(1)
                    continue  # 重新去唤醒识别
                else:
                    logger.info(f"\033[31m唤醒成功！2s后开始识别！\033[0m")
                    time.sleep(2)
                    for i in range(2):  # 每次6s的时间用于识别
                        result = asr_fuc(wav_data)
                        asr_result = asr_result + result
                    chat_result = chatglm_fuc(prompt=asr_result, history_data_list=[],
                                              history_data_len="100")  # 将每6s的录音识别内容 送至大模型进行对话
                    xtts_result = xtts_fuc(str(chat_result), "zh")
                    wavfile.write("./tts_output.wav", 16000, xtts_result)

        else:
            # 唤醒失败，继续唤醒
            kws_result = kws_fuc(wav_data)
            if kws_result is None:
                logger.info(f"\033[31m唤醒失败！1s后重新唤醒！\033[0m")
                time.sleep(1)
            else:
                logger.info("唤醒成功！ 开始识别！")
                for i in range(2):  # 每次6s的时间用于识别
                    result = asr_fuc(wav_data)
                    asr_result = asr_result + result
                chat_result = chatglm_fuc(prompt=asr_result, history_data_list=[],
                                          history_data_len="100")  # 将每6s的录音识别内容 送至大模型进行对话
                xtts_result = xtts_fuc(str(chat_result), "zh")
                wavfile.write("./tts_output.wav", 16000, xtts_result)
