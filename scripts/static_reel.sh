#!/usr/bin/env bash
# Static-crop 9:16 reel (no ball tracking): six clips, hard cuts, 0.5x slow-mo on clip 5, end card on clip 6,
# plus a cover image. Simpler fallback to `ocv-reel`.
#
# Usage: scripts/static_reel.sh SOURCE CLIPS_FILE OUT_MP4 [COVER_JPG]
#   CLIPS_FILE: 6 non-comment lines "start_s dur_s crop_x" (see examples/static_reel_clips.example.txt)
# Env (all optional):
#   TITLE="DOUBLES · 6-4 6-3"  WINNERS="Team A"  LOSERS="Team B"  SCORE="6-4  6-3"
#   COVER_T=<s, default: 1.4 s before the end of clip 4>  FONT=<ttf>  FFMPEG=<path>  VCODEC=<encoder>
set -euo pipefail
[ $# -ge 3 ] || { sed -n '2,10p' "$0"; exit 1; }
SRC=$1; CLIPS=$2; OUT=$3; COVER=${4:-}
TITLE=${TITLE:-}; WINNERS=${WINNERS:-Team A}; LOSERS=${LOSERS:-Team B}; SCORE=${SCORE:-}
FF=${FFMPEG:-$( [ -x "$HOME/bin/ffmpeg" ] && echo "$HOME/bin/ffmpeg" || echo ffmpeg )}
if [ -z "${FONT:-}" ]; then
  for f in "/System/Library/Fonts/Supplemental/Arial Bold.ttf" /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; do
    [ -f "$f" ] && FONT=$f && break
  done
fi
FO=""; [ -n "${FONT:-}" ] && FO="fontfile='$FONT':"
if [ -z "${VCODEC:-}" ]; then [ "$(uname)" = Darwin ] && VCODEC=h264_videotoolbox || VCODEC=libx264; fi

ST=(); DU=(); CX=()
while read -r s d x _; do
  case "$s" in ''|\#*) continue ;; esac
  ST+=("$s"); DU+=("$d"); CX+=("$x")
done < "$CLIPS"
[ ${#ST[@]} -eq 6 ] || { echo "need exactly 6 clips in $CLIPS, got ${#ST[@]}" >&2; exit 1; }

V="crop=1215:2160:X:0,scale=1080:1920:flags=lanczos,setsar=1,fps=60"
v() { echo "${V/X/$1}"; }
IN=()
for i in 0 1 2 3 4 5; do IN+=(-ss "${ST[$i]}" -t "${DU[$i]}" -i "$SRC"); done
TITLE_F=""
[ -n "$TITLE" ] && TITLE_F=",drawtext=${FO}text='$TITLE':fontsize=92:fontcolor=white:borderw=5:bordercolor=black@0.7:box=1:boxcolor=black@0.45:boxborderw=24:x=(w-tw)/2:y=330:enable='lt(t,1.5)'"

nice -n 10 "$FF" -nostdin -hide_banner -loglevel error -y "${IN[@]}" \
 -filter_complex "
 [0:v]$(v "${CX[0]}")${TITLE_F}[v0];
 [1:v]$(v "${CX[1]}")[v1]; [2:v]$(v "${CX[2]}")[v2]; [3:v]$(v "${CX[3]}")[v3];
 [4:v]$(v "${CX[4]}"),setpts=2.0*PTS,fps=60[v4];
 [5:v]$(v "${CX[5]}"),drawbox=x=0:y=0:w=iw:h=ih:color=black@0.45:t=fill:enable='gte(t,0.3)',
   drawtext=${FO}text='$WINNERS':fontsize=96:fontcolor=white:x=(w-tw)/2:y=780:enable='gte(t,0.3)',
   drawtext=${FO}text='def.':fontsize=64:fontcolor=0xF2C14E:x=(w-tw)/2:y=905:enable='gte(t,0.3)',
   drawtext=${FO}text='$LOSERS':fontsize=96:fontcolor=white:x=(w-tw)/2:y=1000:enable='gte(t,0.3)',
   drawtext=${FO}text='${SCORE:- }':fontsize=72:fontcolor=white@0.85:x=(w-tw)/2:y=1150:enable='gte(t,0.3)'[v5];
 [0:a]aresample=48000,afade=t=in:d=0.05[a0]; [1:a]aresample=48000[a1]; [2:a]aresample=48000[a2]; [3:a]aresample=48000[a3];
 [4:a]aresample=48000,atempo=0.5[a4]; [5:a]aresample=48000,afade=t=out:st=2.0:d=0.5[a5];
 [v0][a0][v1][a1][v2][a2][v3][a3][v4][a4][v5][a5]concat=n=6:v=1:a=1[vv][aa];
 [aa]loudnorm=I=-14:TP=-1.5:LRA=11,volume=2dB,alimiter=limit=0.7:level=false,aresample=48000[ao]" \
 -map "[vv]" -map "[ao]" -r 60 -c:v "$VCODEC" -b:v 15M -maxrate 18M -bufsize 30M -profile:v high -pix_fmt yuv420p \
 -c:a aac -b:a 192k -ar 48000 -movflags +faststart "$OUT"

if [ -n "$COVER" ]; then
  # cover: frame from the build-up clip + title
  CT=${COVER_T:-$(awk -v s="${ST[3]}" -v d="${DU[3]}" 'BEGIN{printf "%.2f", s + d - 1.4}')}
  nice -n 10 "$FF" -nostdin -hide_banner -loglevel error -y -ss "$CT" -i "$SRC" -frames:v 1 \
    -vf "$(v "${CX[3]}"),drawbox=x=0:y=280:w=iw:h=260:color=black@0.35:t=fill,drawtext=${FO}text='${TITLE:- }':fontsize=96:fontcolor=white:x=(w-tw)/2:y=360,drawtext=${FO}text='$WINNERS def. $LOSERS':fontsize=54:fontcolor=white:borderw=4:bordercolor=black@0.6:x=(w-tw)/2:y=1440" \
    -q:v 2 "$COVER"
fi
echo done
