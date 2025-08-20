import soundfile  # Import soundfile before torchaudio to register backend
import torch
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import numpy as np
import time
import streamlit as st

# ========== Load Model ==========
model_path = r"mosquito_cnn_final_torchscript.pt"
model = torch.jit.load(model_path, map_location="cpu")
model.eval()

# Species index mapping
species_index_map = {
    0: 'no_mosquito', 1: 'ae aegypti', 2: 'ae albopictus', 3: 'an albimanus',
    4: 'an arabiensis', 5: 'an atroparvus', 6: 'an barbirostris', 7: 'an coluzzii',
    8: 'an coustani', 9: 'an dirus', 10: 'an farauti', 11: 'an freeborni',
    12: 'an funestus', 13: 'an funestus sl', 14: 'an funestus ss', 15: 'an gambiae',
    16: 'an gambiae sl', 17: 'an gambiae ss', 18: 'an harrisoni', 19: 'an leesoni',
    20: 'an maculatus', 21: 'an maculipalpis', 22: 'an merus', 23: 'an minimus',
    24: 'an pharoensis', 25: 'an quadriannulatus', 26: 'an rivulorum', 27: 'an sinensis',
    28: 'an squamosus', 29: 'an stephensi', 30: 'an ziemanni', 31: 'coquillettidia sp',
    32: 'culex pipiens complex', 33: 'culex quinquefasciatus', 34: 'culex tarsalis',
    35: 'culex tigripes', 36: 'ma africanus', 37: 'ma uniformis', 38: 'toxorhynchites brevipalpis'
}

# Species information mapping (shortened here for space, keep your full mapping)
species_info = {
    'ae aegypti': {
        'danger': 'HIGH',
        'diseases': ['Dengue', 'Zika', 'Chikungunya', 'Yellow Fever'],
        'solutions': ['Eliminate standing water in containers', 'Use repellents (DEET, picaridin)', 'Insecticide-treated nets', 'Larvicides (Bti)', 'Genetically modified mosquitoes', 'Indoor residual spraying']
    },
    'ae albopictus': {
        'danger': 'HIGH',
        'diseases': ['Dengue', 'Zika', 'Chikungunya', 'Dirofilariasis'],
        'solutions': ['Source reduction for breeding sites', 'Use repellents', 'Mosquito traps', 'Larvicides', 'Community education', 'Indoor residual spraying']
    },
    'an albimanus': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an arabiensis': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an atroparvus': {
        'danger': 'MEDIUM',
        'diseases': ['Malaria (historical)'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an barbirostris': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an coluzzii': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an coustani': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an dirus': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an farauti': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an freeborni': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an funestus': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an funestus sl': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an funestus ss': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an gambiae': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an gambiae sl': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an gambiae ss': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an harrisoni': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an leesoni': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an maculatus': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an maculipalpis': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an merus': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an minimus': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an pharoensis': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an quadriannulatus': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an rivulorum': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an sinensis': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an squamosus': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an stephensi': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'an ziemanni': {
        'danger': 'HIGH',
        'diseases': ['Malaria'],
        'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
    },
    'coquillettidia sp': {
        'danger': 'MEDIUM',
        'diseases': ['West Nile Virus', 'Rift Valley Fever', 'Eastern Equine Encephalitis'],
        'solutions': ['Vegetation management in wetlands', 'Larvicides', 'Repellents', 'Insecticide sprays']
    },
    'culex pipiens complex': {
        'danger': 'MEDIUM',
        'diseases': ['West Nile Virus', 'Lymphatic Filariasis', 'Japanese Encephalitis'],
        'solutions': ['Eliminate stagnant water', 'Larvicides', 'Adulticides', 'Repellents', 'Bird control for WNV']
    },
    'culex quinquefasciatus': {
        'danger': 'MEDIUM',
        'diseases': ['West Nile Virus', 'Lymphatic Filariasis', 'St. Louis Encephalitis'],
        'solutions': ['Eliminate stagnant water', 'Larvicides', 'Adulticides', 'Repellents']
    },
    'culex tarsalis': {
        'danger': 'MEDIUM',
        'diseases': ['West Nile Virus', 'St. Louis Encephalitis', 'Western Equine Encephalitis'],
        'solutions': ['Eliminate stagnant water', 'Larvicides', 'Adulticides', 'Repellents']
    },
    'culex tigripes': {
        'danger': 'MEDIUM',
        'diseases': ['Lymphatic Filariasis'],
        'solutions': ['Eliminate stagnant water', 'Larvicides', 'Adulticides', 'Repellents']
    },
    'ma africanus': {
        'danger': 'MEDIUM',
        'diseases': ['Lymphatic Filariasis'],
        'solutions': ['Clear aquatic vegetation', 'Larvicides', 'Mass drug administration for filariasis', 'Repellents']
    },
    'ma uniformis': {
        'danger': 'MEDIUM',
        'diseases': ['Lymphatic Filariasis'],
        'solutions': ['Clear aquatic vegetation', 'Larvicides', 'Mass drug administration for filariasis', 'Repellents']
    },
    'toxorhynchites brevipalpis': {
        'danger': 'LOW',
        'diseases': ['None (beneficial, preys on other mosquito larvae)'],
        'solutions': ['Encourage presence for natural biological control']
    }
}

# ========== Audio Recording Settings ==========
SAMPLE_RATE = 16000   # Hz
DURATION = 1          # seconds (record in chunks of 1 second)

# ========== Transform to Spectrogram ==========
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=64,
    n_fft=1024,
    hop_length=512
)

def predict_from_audio(audio_chunk):
    """Convert raw audio -> mel spectrogram -> model prediction"""
    audio_tensor = torch.from_numpy(audio_chunk).float()

    if audio_tensor.ndim > 1:  # stereo → mono
        audio_tensor = audio_tensor.mean(dim=1)

    spec = transform(audio_tensor)

    spec = torch.nn.functional.interpolate(spec.unsqueeze(0).unsqueeze(0),
                                           size=(64, 64),
                                           mode="bilinear",
                                           align_corners=False)

    with torch.no_grad():
        output = model(spec)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    return predicted_class, probs[0, predicted_class].item()

# ========== Streamlit App ==========
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        border-radius: 8px;
    }
    .detection {
        font-size: 26px;
        font-weight: bold;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🦟 Mosquito Detector 🦟</h1>", unsafe_allow_html=True)

# Initialize session state
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'latest_detection' not in st.session_state:
    st.session_state.latest_detection = "Not detecting yet... 🎤"
if 'latest_info' not in st.session_state:
    st.session_state.latest_info = ""

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Listening 🎙️"):
        st.session_state.listening = True

with col2:
    if st.button("Stop Listening 🛑"):
        st.session_state.listening = False

status = "Listening... 🔊" if st.session_state.listening else "Stopped ❌"
st.markdown(f"<p style='text-align: center; font-size: 20px;'>{status}</p>", unsafe_allow_html=True)

# Displays
detection_display = st.empty()
info_display = st.empty()

# Always show the last known detection + info
detection_display.markdown(f"<p class='detection' style='text-align: center;'>{st.session_state.latest_detection}</p>", unsafe_allow_html=True)
if st.session_state.latest_info:
    info_display.markdown(st.session_state.latest_info, unsafe_allow_html=True)

# Process live audio only when listening
if st.session_state.listening:
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)

    pred_class, confidence = predict_from_audio(audio)
    species_name = species_index_map[pred_class]
    detection = f"Detected: {species_name} (confidence: {confidence:.2f}) 🦟"

    # ✅ Update only when a new species is found
    if detection != st.session_state.latest_detection:
        st.session_state.latest_detection = detection

        if pred_class != 0:  # ignore "no_mosquito"
            info = species_info.get(species_name, {})
            if info:
                st.session_state.latest_info = f"""
                <h3 style='text-align: center;'>Species Information</h3>
                <p><strong>Danger Level:</strong> {info['danger']}</p>
                <p><strong>Diseases Spread:</strong> {', '.join(info['diseases'])}</p>
                <p><strong>Solutions to Control:</strong></p>
                <ul>
                {"".join(f"<li>{sol}</li>" for sol in info['solutions'])}
                </ul>
                """

    time.sleep(0.01)
    st.rerun()






















# # Species information mapping (shortened here for space, keep your full mapping)
# species_info = {
#     'ae aegypti': {
#         'danger': 'HIGH',
#         'diseases': ['Dengue', 'Zika', 'Chikungunya', 'Yellow Fever'],
#         'solutions': ['Eliminate standing water in containers', 'Use repellents (DEET, picaridin)', 'Insecticide-treated nets', 'Larvicides (Bti)', 'Genetically modified mosquitoes', 'Indoor residual spraying']
#     },
#     'ae albopictus': {
#         'danger': 'HIGH',
#         'diseases': ['Dengue', 'Zika', 'Chikungunya', 'Dirofilariasis'],
#         'solutions': ['Source reduction for breeding sites', 'Use repellents', 'Mosquito traps', 'Larvicides', 'Community education', 'Indoor residual spraying']
#     },
#     'an albimanus': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an arabiensis': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an atroparvus': {
#         'danger': 'MEDIUM',
#         'diseases': ['Malaria (historical)'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an barbirostris': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an coluzzii': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an coustani': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an dirus': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an farauti': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an freeborni': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an funestus': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an funestus sl': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an funestus ss': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an gambiae': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an gambiae sl': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an gambiae ss': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an harrisoni': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an leesoni': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an maculatus': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an maculipalpis': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an merus': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an minimus': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an pharoensis': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an quadriannulatus': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an rivulorum': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an sinensis': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an squamosus': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an stephensi': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'an ziemanni': {
#         'danger': 'HIGH',
#         'diseases': ['Malaria'],
#         'solutions': ['Insecticide-treated nets (ITNs)', 'Indoor residual spraying (IRS)', 'Larval source management', 'Repellents', 'House screening']
#     },
#     'coquillettidia sp': {
#         'danger': 'MEDIUM',
#         'diseases': ['West Nile Virus', 'Rift Valley Fever', 'Eastern Equine Encephalitis'],
#         'solutions': ['Vegetation management in wetlands', 'Larvicides', 'Repellents', 'Insecticide sprays']
#     },
#     'culex pipiens complex': {
#         'danger': 'MEDIUM',
#         'diseases': ['West Nile Virus', 'Lymphatic Filariasis', 'Japanese Encephalitis'],
#         'solutions': ['Eliminate stagnant water', 'Larvicides', 'Adulticides', 'Repellents', 'Bird control for WNV']
#     },
#     'culex quinquefasciatus': {
#         'danger': 'MEDIUM',
#         'diseases': ['West Nile Virus', 'Lymphatic Filariasis', 'St. Louis Encephalitis'],
#         'solutions': ['Eliminate stagnant water', 'Larvicides', 'Adulticides', 'Repellents']
#     },
#     'culex tarsalis': {
#         'danger': 'MEDIUM',
#         'diseases': ['West Nile Virus', 'St. Louis Encephalitis', 'Western Equine Encephalitis'],
#         'solutions': ['Eliminate stagnant water', 'Larvicides', 'Adulticides', 'Repellents']
#     },
#     'culex tigripes': {
#         'danger': 'MEDIUM',
#         'diseases': ['Lymphatic Filariasis'],
#         'solutions': ['Eliminate stagnant water', 'Larvicides', 'Adulticides', 'Repellents']
#     },
#     'ma africanus': {
#         'danger': 'MEDIUM',
#         'diseases': ['Lymphatic Filariasis'],
#         'solutions': ['Clear aquatic vegetation', 'Larvicides', 'Mass drug administration for filariasis', 'Repellents']
#     },
#     'ma uniformis': {
#         'danger': 'MEDIUM',
#         'diseases': ['Lymphatic Filariasis'],
#         'solutions': ['Clear aquatic vegetation', 'Larvicides', 'Mass drug administration for filariasis', 'Repellents']
#     },
#     'toxorhynchites brevipalpis': {
#         'danger': 'LOW',
#         'diseases': ['None (beneficial, preys on other mosquito larvae)'],
#         'solutions': ['Encourage presence for natural biological control']
#     }
# }