# train_watcher.py
import os
import time
import subprocess

inter_data_path = '/mnt/zhouting/tpt_remove_allow/grpo1/'


SIGNAL_PATH = os.path.join(inter_data_path, "start_hidden_classifier_train.signal")

while True:
    if os.path.exists(SIGNAL_PATH):
        with open(SIGNAL_PATH) as f:
            sh_path = f.read().strip()

        os.remove(SIGNAL_PATH)

        print(f"[watcher] Signal received, executing sh: {sh_path}")

        subprocess.Popen(
            [sh_path],
            start_new_session=True,
        )

    time.sleep(1)