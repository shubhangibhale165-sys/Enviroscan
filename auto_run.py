import time
import subprocess

# ⏱ Time interval (in seconds)
INTERVAL = 3600   # 1 hour (change if needed)

print(" Auto Runner Started...\n")

while True:
    print("▶ Running main.py...\n")

    try:
        subprocess.run(["python", "main.py"], check=True)
        print(" Run completed successfully\n")

    except subprocess.CalledProcessError:
        print(" Error occurred during execution\n")

    print(f" Waiting for {INTERVAL} seconds...\n")
    time.sleep(INTERVAL)