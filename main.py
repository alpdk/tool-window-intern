import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def clean_data(df, mode):
    episodes = []

    for user, group in df.groupby("user_id"):
        current_open = None

        for a, row in group.iterrows():
            event, ts, _ = row["event"], row["timestamp"], row.get("open_type", None)

            if event == "opened":
                if mode == "fifo":
                    if current_open is None:
                        current_open = row
                elif mode == "lifo":
                    current_open = row
                elif mode == "switch_split":
                    if current_open is not None:
                        duration = (ts - current_open["timestamp"]) / 1000
                        episodes.append({
                            "user_id": user,
                            "open_type": current_open["open_type"],
                            "start_ts": current_open["timestamp"],
                            "end_ts": ts,
                            "duration_sec": duration
                        })

                    current_open = row
            elif event == "closed":
                if current_open is not None:
                    duration = (ts - current_open["timestamp"]) / 1000
                    episodes.append({
                        "user_id": user,
                        "open_type": current_open["open_type"],
                        "start_ts": current_open["timestamp"],
                        "end_ts": ts,
                        "duration_sec": duration
                    })
                current_open = None

    return pd.DataFrame(episodes)

if __name__ == '__main__':
    df = pd.read_csv("toolwindow_data.csv")

    print(df.head())

    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    print(df.head())

    for mode in ["fifo", "lifo", "switch_split"]:
        print(f"\n=== Mode: {mode} ===")

        clean_df = clean_data(df, mode)

        summary = clean_df.groupby("open_type")["duration_sec"].describe()
        print("Summary statistics:")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(summary)
        pd.reset_option("display.max_columns")
        pd.reset_option("display.width")

        # --- Boxplot ---
        manual_durations = clean_df[clean_df["open_type"] == "manual"]["duration_sec"]
        auto_durations = clean_df[clean_df["open_type"] == "auto"]["duration_sec"]

        plt.figure(figsize=(8, 5))
        plt.boxplot([manual_durations, auto_durations], tick_labels=["Manual", "Auto"])
        plt.ylabel("Duration (seconds)")
        plt.title("Tool Window Duration by Open Type")
        plt.savefig(f"{mode}_toolwindow_boxplot.png", dpi=300, bbox_inches='tight')
        plt.show()

        # --- Histogram ---
        plt.figure(figsize=(8, 5))
        plt.hist(manual_durations, bins=30, alpha=0.6, label="Manual", log=True)
        plt.hist(auto_durations, bins=30, alpha=0.6, label="Auto", log=True)
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Count (log scale)")
        plt.title("Distribution of Tool Window Durations")
        plt.legend()
        plt.savefig(f"{mode}_toolwindow_histogram.png", dpi=300, bbox_inches='tight')
        plt.show()

        stat, p = mannwhitneyu(manual_durations, auto_durations, alternative="two-sided")
        median_diff = manual_durations.median() - auto_durations.median()

        print(f"Mann-Whitney U test: statistic={stat:.2f}, p-value={p:.4f}")
        print(f"Median difference (manual - auto): {median_diff:.2f} seconds")