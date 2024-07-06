#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script performs speaker diarization task based on audio-only input, 
# in contrast to "run_video.sh" which is based on video and audio input.

set -e
. ./path.sh || exit 1

stage=1
stop_stage=6

wav_list=examples/wav.list
exp=exp
conf_file=conf/diar.yaml
gpus="0 1 2 3"
nj=8

. local/parse_options.sh || exit 1

json_dir=$exp/json
embs_dir=$exp/embs
rttm_dir=$exp/rttm

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
  echo "$(basename $0) Stage 1: Prepare input wavs..."
  mkdir -p examples
  wget "https://modelscope.cn/api/v1/models/damo/speech_eres2net-large_speaker-diarization_common/repo\
?Revision=master&FilePath=examples/2speakers_example.wav" -O examples/2speakers_example.wav
  wget "https://modelscope.cn/api/v1/models/damo/speech_eres2net-large_speaker-diarization_common/repo\
?Revision=master&FilePath=examples/2speakers_example.rttm" -O examples/2speakers_example.rttm
  echo "examples/2speakers_example.wav" > examples/wav.list
  echo "examples/2speakers_example.rttm" > examples/refrttm.list
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "$(basename $0) Stage2: Do vad for input wavs..."
  python local/voice_activity_detection.py --wavs $wav_list --out_file $json_dir/vad.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "$(basename $0) Stage3: Prepare subsegments info..."
  python local/prepare_subseg_json.py --vad $json_dir/vad.json --out_file $json_dir/subseg.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "$(basename $0) Stage4: Extract speaker embeddings..."
  # Set speaker_model_id to damo/speech_eres2net_sv_zh-cn_16k-common when using eres2net 
  speaker_model_id=damo/speech_campplus_sv_zh-cn_16k-common
  torchrun --nproc_per_node=$nj local/extract_diar_embeddings.py --model_id $speaker_model_id --conf $conf_file \
          --subseg_json $json_dir/subseg.json --embs_out $embs_dir --gpu $gpus --use_gpu
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "$(basename $0) Stage5: Perform clustering and output sys rttms..."
  torchrun --nproc_per_node=$nj local/cluster_and_postprocess.py --conf $conf_file --wavs $wav_list \
          --audio_embs_dir $embs_dir --rttm_dir $rttm_dir
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "$(basename $0) Stage6: Get the final metrics..."
  ref_rttm_list=examples/refrttm.list
  if [ -f $ref_rttm_list ]; then
    cat $ref_rttm_list | while read line;do cat $line;done > $exp/concat_ref_rttm
    echo "Computing DER..."
    python local/compute_der.py --exp_dir $exp --ref_rttm $exp/concat_ref_rttm
  else
    echo "Refrttm.list is not detected. Can't calculate the result"
  fi
fi
