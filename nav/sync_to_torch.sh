#!/bin/bash
# Back up results from 127 to Torch. Runs on Torch's data-transfer node (dtn011), pulling from 127
# with the dedicated key ~/.ssh/id_lewm_127. Trigger from a machine that has a live `ssh torch` session:
#
#   ssh torch 'ssh -o HostKeyAlias=dtn.torch.hpc.nyu.edu dtn011.hpc.nyu.edu \
#              "bash /scratch/yf2685/lewm-nav/le-wm/nav/sync_to_torch.sh [what...]"'
#
# what: any of  runs reports logs h5   (default: runs reports logs; h5 only when asked)
set -u
SRC=nyuair@10.224.36.127
SSH="ssh -o BatchMode=yes -i $HOME/.ssh/id_lewm_127"
DST=/scratch/yf2685/lewm-nav
WHAT=${*:-runs reports logs}
for w in $WHAT; do
  case $w in
    runs|reports|logs) from=/home/nyuair/yuxuan/lewm-nav/$w/ ;;
    h5) from=/home/nyuair/data-001/yuxuan_lewm/h5/ ;;
    *) echo "unknown: $w"; continue ;;
  esac
  mkdir -p "$DST/$w"
  echo "== sync $w"
  rsync -a --partial --exclude '*.part' -e "$SSH" "$SRC:$from" "$DST/$w/" && echo "ok $w $(du -sh $DST/$w | cut -f1)"
done
date > "$DST/LAST_SYNC"
