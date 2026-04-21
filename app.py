import os
import time
import base64
from io import BytesIO
import wave
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit.components.v1 as components


# =========================
# Page setup
# =========================
st.set_page_config(layout="wide", page_title="🚨Drilling IDSS🚨")

st.title("🚨 AI Drilling Early Warning System")
st.markdown("### Intelligent Decision Support System (IDSS) for Stuck Pipe Prevention")
st.markdown("---")


# =========================
# Session state
# =========================
defaults = {
    "muted_alarm_level": None,   # None / orange / red
    "audio_unlocked": False,
    "idx": 100,
    "model": None,
    "features": None,
    "displayed_alarm_level": None,
    "active_alarm": None,
    "alarm_history": [],
    "alarm_counter": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================
# Sidebar controls
# =========================
st.sidebar.subheader("Controls")

stop_pressed = st.sidebar.button("🔇 Stop current alarm")
enable_sound_pressed = st.sidebar.button("🔊 Enable sound once")

if enable_sound_pressed:
    st.session_state.audio_unlocked = True

st.sidebar.caption("If your browser blocks autoplay audio, click 'Enable sound once' before the demo.")

st.sidebar.markdown("---")
st.sidebar.subheader("Color Threshold Controls")

green_z_limit = st.sidebar.number_input("Green z-score", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
green_ratio_limit = st.sidebar.number_input("Green ratio", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
orange_z_limit = st.sidebar.number_input("Orange z-score", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
orange_rpm_limit = st.sidebar.number_input("Orange min RPM", min_value=0.0, max_value=500.0, value=60.0, step=1.0)

refresh_seconds = st.sidebar.number_input(
    "Refresh interval (seconds)",
    min_value=0.05,
    max_value=1.0,
    value=0.20,
    step=0.01
)


# =========================
# Audio helpers
# =========================
def generate_tone(frequency=1000, duration=0.5, volume=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (volume * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    buffer.seek(0)
    return buffer.read()


def audio_data_uri(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


# Orange: lighter
ORANGE_SRC = audio_data_uri(generate_tone(frequency=850, duration=0.30, volume=0.25))

# Red: stronger
RED_SRC = audio_data_uri(generate_tone(frequency=1800, duration=1.00, volume=1.00))


def render_alarm_controller(mode: str):
    """
    mode: 'stop' | 'orange' | 'red'
    Creates/updates ONE audio element in the parent document.
    """
    if mode == "orange":
        src = ORANGE_SRC
        volume = 0.25
    elif mode == "red":
        src = RED_SRC
        volume = 1.00
    else:
        src = ""
        volume = 0.0

    js = f"""
    <script>
    (function() {{
        const doc = window.parent.document;

        // Remove duplicated audios if any
        const duplicates = doc.querySelectorAll('audio[data-drilling-alarm="1"]');
        duplicates.forEach((a, idx) => {{
            if (idx > 0) {{
                try {{ a.pause(); }} catch(e) {{}}
                try {{ a.remove(); }} catch(e) {{}}
            }}
        }});

        let audio = doc.getElementById("drilling-global-alarm");
        if (!audio) {{
            audio = doc.createElement("audio");
            audio.id = "drilling-global-alarm";
            audio.setAttribute("data-drilling-alarm", "1");
            audio.loop = true;
            audio.style.display = "none";
            doc.body.appendChild(audio);
        }}

        const mode = "{mode}";
        const desiredSrc = "{src}";
        const desiredVolume = {volume};

        if (mode === "stop") {{
            try {{ audio.pause(); }} catch(e) {{}}
            audio.removeAttribute("src");
            audio.load();
            audio.dataset.mode = "stop";
            return;
        }}

        if (audio.dataset.mode !== mode || audio.getAttribute("src") !== desiredSrc) {{
            try {{ audio.pause(); }} catch(e) {{}}
            audio.setAttribute("src", desiredSrc);
            audio.volume = desiredVolume;
            audio.dataset.mode = mode;
            audio.load();
        }} else {{
            audio.volume = desiredVolume;
        }}

        const playPromise = audio.play();
        if (playPromise) {{
            playPromise.catch(() => {{}});
        }}
    }})();
    </script>
    """
    components.html(js, height=0, width=0)


# =========================
# Alarm helpers
# =========================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def start_alarm(level, idx, row):
    st.session_state.alarm_counter += 1
    st.session_state.active_alarm = {
        "Alarm ID": st.session_state.alarm_counter,
        "Level": level.upper(),
        "Start Index": int(idx),
        "End Index": None,
        "Start Time": now_str(),
        "End Time": None,
        "Duration (samples)": 1,
        "Max Torque": float(row["Torque"]),
        "Max z-score": float(row["z"]),
        "Status": "Active",
    }


def update_alarm(idx, row):
    if st.session_state.active_alarm is None:
        return

    st.session_state.active_alarm["End Index"] = int(idx)
    st.session_state.active_alarm["Duration (samples)"] = int(idx) - int(st.session_state.active_alarm["Start Index"]) + 1
    st.session_state.active_alarm["Max Torque"] = max(float(st.session_state.active_alarm["Max Torque"]), float(row["Torque"]))
    st.session_state.active_alarm["Max z-score"] = max(float(st.session_state.active_alarm["Max z-score"]), float(row["z"]))


def close_alarm(idx):
    if st.session_state.active_alarm is None:
        return

    st.session_state.active_alarm["End Index"] = int(idx)
    st.session_state.active_alarm["End Time"] = now_str()
    st.session_state.active_alarm["Status"] = "Cleared"

    st.session_state.alarm_history.append(st.session_state.active_alarm.copy())
    st.session_state.active_alarm = None


def get_alarm_history_df():
    if len(st.session_state.alarm_history) == 0:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.alarm_history)


# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "rigsense Data.csv")

    df = pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="skip", skiprows=1)
    df.columns = df.columns.str.strip()

    df["Torque"] = pd.to_numeric(df["TOP:2"], errors="coerce").ffill().fillna(0)
    df["RPM"] = pd.to_numeric(df["ROTA:2"], errors="coerce").ffill().fillna(0)
    df["HOOK"] = pd.to_numeric(df["HOOK"], errors="coerce").ffill().fillna(0)
    df["FLOW"] = pd.to_numeric(df["FLOW:1"], errors="coerce").ffill().fillna(0)

    return df


try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ File 'rigsense Data.csv' not found in the same folder as this script.")
    st.stop()


# =========================
# Feature engineering
# =========================
df["Torque_diff"] = df["Torque"].diff().fillna(0)
df["ratio"] = df["Torque"] / (df["RPM"] + 1)
df["flow_diff"] = df["FLOW"].diff().fillna(0)
df["hook_diff"] = df["HOOK"].diff().fillna(0)

window = 30
df["mean"] = df["Torque"].rolling(window).mean()
df["std"] = df["Torque"].rolling(window).std().replace(0, np.nan)
df["z"] = ((df["Torque"] - df["mean"]) / df["std"]).replace([np.inf, -np.inf], np.nan).fillna(0)


# =========================
# Rule-based decision
# =========================
def classify(z, ratio, rpm, flow_diff):
    if z < green_z_limit and ratio < green_ratio_limit:
        return "🟢 GREEN (Normal / Early Warning)", "Normal operation. The process is stable and within normal limits."
    elif z < orange_z_limit or rpm < orange_rpm_limit:
        return "🟠 ORANGE (Warning / Pre-Sticking)", "Early warning. An abnormal trend is detected and preventive action is recommended."
    else:
        return "🔴 RED (CRITICAL / STUCK PIPE)", "Critical condition. High risk of stuck pipe. Immediate action is required."


# =========================
# ML model
# =========================
df["future_torque"] = df["Torque"].shift(-30)
df["label"] = (df["future_torque"] > df["Torque"] * 1.3).astype(int)

features = ["Torque", "RPM", "Torque_diff", "ratio", "flow_diff", "hook_diff"]
df_ml = df.dropna(subset=features + ["label"]).copy()

if len(df_ml) < 10:
    st.error("❌ Not enough data to train the AI model.")
    st.stop()

if st.session_state.model is None:
    X = df_ml[features]
    y = df_ml["label"]
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    st.session_state.model = model
    st.session_state.features = features

model = st.session_state.model
features = st.session_state.features


# =========================
# End-of-data behavior
# =========================
if st.session_state.idx >= len(df):
    if st.session_state.active_alarm is not None:
        close_alarm(len(df) - 1)

    st.session_state.idx = 100
    st.session_state.muted_alarm_level = None
    st.session_state.displayed_alarm_level = None


# =========================
# Current sample
# =========================
i = st.session_state.idx
row = df.iloc[i]

status_color, action_text = classify(row["z"], row["ratio"], row["RPM"], row["flow_diff"])

X_live = pd.DataFrame([[
    row["Torque"], row["RPM"], row["Torque_diff"],
    row["ratio"], row["flow_diff"], row["hook_diff"]
]], columns=features)

prediction = model.predict(X_live)[0]

current_alarm_level = None
if "ORANGE" in status_color:
    current_alarm_level = "orange"
elif "RED" in status_color:
    current_alarm_level = "red"

st.session_state.displayed_alarm_level = current_alarm_level

# Stop current alarm فقط للون الحالي
if stop_pressed and current_alarm_level in ["orange", "red"]:
    st.session_state.muted_alarm_level = current_alarm_level


# =========================
# Alarm state machine
# =========================
if current_alarm_level is None:
    if st.session_state.active_alarm is not None:
        close_alarm(i)
    st.session_state.muted_alarm_level = None
else:
    if st.session_state.active_alarm is None:
        start_alarm(current_alarm_level, i, row)
    else:
        active_level = st.session_state.active_alarm["Level"].lower()
        if active_level != current_alarm_level:
            close_alarm(i - 1 if i > 0 else i)
            start_alarm(current_alarm_level, i, row)

        update_alarm(i, row)


# =========================
# Decide alarm audio command
# =========================
if current_alarm_level is None:
    alarm_mode = "stop"
elif not st.session_state.audio_unlocked:
    alarm_mode = "stop"
elif st.session_state.muted_alarm_level == current_alarm_level:
    alarm_mode = "stop"
else:
    alarm_mode = current_alarm_level

# عنصر صوت واحد فقط
render_alarm_controller(alarm_mode)


# =========================
# Tabs
# =========================
dashboard_tab, alarm_tab = st.tabs(["📡 Dashboard", "🚨 Alarm Page"])

with dashboard_tab:
    st.subheader("📡 Live Monitoring Dashboard")

    if "GREEN" in status_color:
        st.success(f"**Status:** {status_color}\n\n**Action:** {action_text}")
    elif "ORANGE" in status_color:
        st.warning(f"**Status:** {status_color}\n\n**Action:** {action_text}")
    else:
        st.error(f"**Status:** {status_color}\n\n**Action:** {action_text}")

    # خانة AI Prediction ثابتة دائمًا
    st.markdown("### AI Prediction")
    if prediction == 1:
        st.info("🤖 High probability of abnormal torque increase shortly. Prepare to mitigate!")
    else:
        st.info("🤖 Normal trend. No abnormal torque increase predicted.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Torque (TOP:2)", round(row["Torque"], 2), round(row["Torque_diff"], 2))
    col2.metric("Hook Load", round(row["HOOK"], 2), round(row["hook_diff"], 2))
    col3.metric("RPM", round(row["RPM"], 2))
    col4.metric("Flow Rate", round(row["FLOW"], 2), round(row["flow_diff"], 2))

    start_idx = max(0, i - 100)
    chart_data = df.iloc[start_idx:i][["Torque", "HOOK"]].copy()
    chart_data["HOOK"] = chart_data["HOOK"] / 10.0
    st.line_chart(chart_data, use_container_width=True)

    st.caption(f"Current index: {i} / {len(df) - 1}")

with alarm_tab:
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🗑️ Clear Alarm History", key="clear_alarm_history_btn"):
            st.session_state.alarm_history = []
            st.rerun()

    with c2:
        hist_df_for_download = get_alarm_history_df()
        if hist_df_for_download.empty:
            st.download_button(
                "⬇️ Download Alarm History CSV",
                data="",
                file_name="alarm_history.csv",
                mime="text/csv",
                disabled=True,
                key="download_alarm_history_btn_disabled"
            )
        else:
            csv_data = hist_df_for_download.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Alarm History CSV",
                data=csv_data,
                file_name="alarm_history.csv",
                mime="text/csv",
                key="download_alarm_history_btn"
            )

    st.markdown("### Active Alarm")
    if st.session_state.active_alarm is None:
        st.info("No active alarm.")
    else:
        active_df = pd.DataFrame([st.session_state.active_alarm])
        st.dataframe(active_df, use_container_width=True)

    st.markdown("### All Alarms Summary")
    total_hist = len(st.session_state.alarm_history)
    orange_hist = sum(1 for a in st.session_state.alarm_history if a["Level"] == "ORANGE")
    red_hist = sum(1 for a in st.session_state.alarm_history if a["Level"] == "RED")

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Historical Alarms", total_hist)
    s2.metric("Historical Orange", orange_hist)
    s3.metric("Historical Red", red_hist)

    st.markdown("### Alarm History")
    if len(st.session_state.alarm_history) == 0:
        st.info("Alarm history is empty.")
    else:
        hist_df = pd.DataFrame(st.session_state.alarm_history[::-1])
        st.dataframe(hist_df, use_container_width=True)


# =========================
# Advance and rerun
# =========================
st.session_state.idx += 1
time.sleep(refresh_seconds)
st.rerun()